#pragma once

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFileInfo>
#include <algorithm>

namespace femviewer {
// Directory opening includes nested example folders; opening one file only lists siblings.
inline QFileInfoList resultFiles(const QDir& directory, bool recursive = true) {
    QFileInfoList files;
    QDirIterator iterator(directory.absolutePath(),
        {QStringLiteral("*.h5"), QStringLiteral("*.hdf5")},
        QDir::Files | QDir::Readable,
        recursive ? QDirIterator::Subdirectories : QDirIterator::NoIteratorFlags);
    while (iterator.hasNext()) {
        iterator.next();
        files.append(iterator.fileInfo());
    }
    std::sort(files.begin(), files.end(), [](const QFileInfo& a, const QFileInfo& b) {
        return a.absoluteFilePath().compare(b.absoluteFilePath(), Qt::CaseInsensitive) < 0;
    });
    return files;
}

inline QString defaultResultsDirectory(const QString& family) {
    // Support launching from the repository, a solver folder, or a native build.
    for (const auto& start : {QDir::currentPath(), QCoreApplication::applicationDirPath()}) {
        QDir directory(start);
        do {
            const auto candidate = directory.filePath(family + QStringLiteral("/outputs"));
            if (QDir(candidate).exists() && !resultFiles(QDir(candidate)).isEmpty()) {
                return candidate;
            }
        } while (directory.cdUp());
    }
    return {};
}
} // namespace femviewer
