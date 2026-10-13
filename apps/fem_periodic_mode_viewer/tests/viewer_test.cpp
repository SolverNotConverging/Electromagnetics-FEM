#include "main_window.hpp"
#include "material_surface.hpp"
#include "h5_reader.hpp"
#include "../../../cmake/ViewerResults.hpp"

#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QEventLoop>
#include <QFile>
#include <QTabWidget>
#include <QTemporaryDir>
#include <QTimer>
#include <QThreadPool>
#ifdef FEM_PERIODIC_MODE_VIEWER_WITH_VTK
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>
#include "vtk_field_view.hpp"
#endif

#include <cstdlib>
#include <functional>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}

void loadAndWait(femperiodic::MainWindow& window, const std::function<void()>& action) {
    QEventLoop loop;
    bool finished = false;
    bool success = false;
    QString error;
    window.setLoadCompletionHandler([&](bool loaded, const QString& message) {
        finished = true;
        success = loaded;
        error = message;
        loop.quit();
    });
    QTimer timer;
    timer.setSingleShot(true);
    QObject::connect(&timer, &QTimer::timeout, &loop, &QEventLoop::quit);
    timer.start(10'000);
    action();
    if (!finished) loop.exec();
    window.setLoadCompletionHandler({});
    require(finished, "Viewer loading timed out");
    if (!success) throw std::runtime_error(error.toStdString());
}
}

int main(int argc, char* argv[]) {
#ifdef FEM_PERIODIC_MODE_VIEWER_WITH_VTK
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());
#endif
    QApplication app(argc, argv);
    if (argc != 3 && argc != 4) return EXIT_FAILURE;
    try {
        // Two tetrahedra share an interior material interface. The volume's
        // exterior surface alone cannot display that interface.
        femperiodic::MeshData mesh;
        mesh.points = {{{0, 0, 0}}, {{1, 0, 0}}, {{0, 1, 0}},
                       {{0, 0, 1}}, {{0, 0, -1}}};
        mesh.cells = {{0, 1, 2, 3}, {0, 2, 1, 4}};
        femperiodic::MaterialState material;
        material.epsilonR = {{{6., 6., 6.}}, {{1., 1., 1.}}};
        material.muR = {{{1., 1., 1.}}, {{1., 1., 1.}}};
        auto faces = femperiodic::materialFaces(mesh, material);
        const auto shared = std::array<std::int64_t, 3>{0, 1, 2};
        require(std::count_if(faces.begin(), faces.end(), [&](const auto& face) {
            return face.vertices == shared && face.owner == 0 && !face.boundary;
        }) == 1, "Interior dielectric interface is missing or duplicated");
        material.epsilonR[1] = material.epsilonR[0];
        // Material samples are quadrature averages and can differ by roundoff.
        material.epsilonR[1][0] += 1e-14;
        faces = femperiodic::materialFaces(mesh, material);
        require(std::none_of(faces.begin(), faces.end(), [&](const auto& face) {
            return face.vertices == shared;
        }), "Same-material tetrahedra expose an internal mesh face");
#ifdef FEM_PERIODIC_MODE_VIEWER_WITH_VTK
        if (argc == 4) {
            const auto archive = femperiodic::H5Reader::loadIndex(argv[3]);
            const auto& selected = archive->cases.front();
            const auto meshData = femperiodic::H5Reader::loadMesh(*archive, selected.meshIndex);
            const auto state = femperiodic::H5Reader::loadMaterialState(
                *archive, selected.materialStateIndex);
            const auto fields = femperiodic::H5Reader::loadModeFields(*archive, 0, 0);
            const auto surfaces = femperiodic::materialFaces(*meshData, *state);
            require(std::any_of(surfaces.begin(), surfaces.end(), [](const auto& face) {
                return face.boundary;
            }), "Excluded conductor surfaces are missing");
            femperiodic::VtkFieldView view;
            view.resize(1000, 800);
            view.setMaterialOnly(true);
            view.show();
            view.setData(meshData, state, fields);
            QEventLoop loop;
            QTimer::singleShot(250, &loop, &QEventLoop::quit);
            loop.exec();
            auto* widget = view.findChild<QVTKOpenGLNativeWidget*>();
            require(widget != nullptr, "Missing material viewport");
            const auto output = qEnvironmentVariable("FEM_MATERIAL_SCREENSHOT");
            if (!output.isEmpty())
                require(widget->grabFramebuffer().save(output), "Cannot save material screenshot");
            std::cout << "Material interfaces and excluded conductor surfaces: PASS\n";
            return EXIT_SUCCESS;
        }
#endif
        QTemporaryDir directory;
        require(directory.isValid(), "Cannot create test folder");
        const auto root = directory.path();
        QDir(root).mkpath(QStringLiteral("unrelated/nested"));
        const auto selected = root + QStringLiteral("/selected.h5");
        require(QFile::copy(QString::fromLocal8Bit(argv[1]), selected), "Cannot copy sweep fixture");
        require(QFile::copy(selected, root + QStringLiteral("/unrelated/nested/other.h5")),
                "Cannot copy nested fixture");
        require(femviewer::resultFiles(QDir(root), false).size() == 1,
                "Opening a file searches unrelated subfolders");
        require(femviewer::resultFiles(QDir(root)).size() == 2,
                "Directory browsing omits nested results");
        femperiodic::MainWindow window;
        window.show();
        auto* cases = window.findChild<QComboBox*>(QStringLiteral("cases"));
        auto* modes = window.findChild<QComboBox*>(QStringLiteral("modes"));
        auto* files = window.findChild<QComboBox*>(QStringLiteral("resultFiles"));
        auto* tabs2D = window.findChild<QTabWidget*>(QStringLiteral("fields2D"));
        auto* tabs3D = window.findChild<QTabWidget*>(QStringLiteral("fields3D"));
        require(cases && modes && files && tabs2D && tabs3D, "Missing viewer controls");
        for (auto* tabs : {tabs2D, tabs3D}) {
            require(tabs->tabText(0) == QStringLiteral("Field")
                        && tabs->tabText(1) == QStringLiteral("Material"), "Incorrect tab order");
        }
        loadAndWait(window, [&] { window.loadPath(std::filesystem::path(selected.toStdWString())); });
        require(files->count() == 1, "Single-file viewer includes unrelated nested results");
        require(cases->count() == 2 && cases->itemText(0).startsWith(QStringLiteral("1 · "))
                    && cases->itemText(1).startsWith(QStringLiteral("2 · ")), "Cases are not one-based");
        require(modes->itemText(0).startsWith(QStringLiteral("1 · ")),
                "Modes are not one-based");
        require(tabs2D->currentIndex() == 0, "2D fields are not shown by default");
        require(cases->minimumContentsLength() >= 18 && modes->minimumContentsLength() >= 40,
                "Dropdowns are too narrow");
        for (int index = 0; index < modes->count(); ++index) {
            require(modes->sizeHint().width() > modes->fontMetrics().horizontalAdvance(modes->itemText(index)),
                    "Mode dropdown truncates its label");
        }
        loadAndWait(window, [&] { cases->setCurrentIndex(1); });
        require(tabs2D->currentIndex() == 0, "Changing case does not show fields");
        loadAndWait(window, [&] { window.loadPath(std::filesystem::path(root.toStdWString())); });
        require(files->count() == 2, "Open directory omits nested results");
        loadAndWait(window, [&] { window.loadPath(std::filesystem::path(argv[2])); });
        require(tabs3D->currentIndex() == 0, "3D fields are not shown by default");
        QThreadPool::globalInstance()->waitForDone();
        std::cout << "Field tabs, one-based selections, readable dropdowns, and scoped result browsing: PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
