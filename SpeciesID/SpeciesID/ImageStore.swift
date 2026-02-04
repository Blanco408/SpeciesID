import UIKit

class ImageStore {
    static let shared = ImageStore()

    private let fileManager = FileManager.default

    private var imagesDirectory: URL {
        let docs = fileManager.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let dir = docs.appendingPathComponent("ObservationImages")
        if !fileManager.fileExists(atPath: dir.path) {
            try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        }
        return dir
    }

    // MARK: - Save Image

    func saveImage(_ image: UIImage) -> String? {
        let filename = UUID().uuidString + ".jpg"
        let path = imagesDirectory.appendingPathComponent(filename)

        guard let data = image.jpegData(compressionQuality: 0.8) else {
            return nil
        }

        do {
            try data.write(to: path)
            return filename
        } catch {
            print("Failed to save image: \(error)")
            return nil
        }
    }

    // MARK: - Load Image

    func loadImage(filename: String) -> UIImage? {
        let path = imagesDirectory.appendingPathComponent(filename)
        return UIImage(contentsOfFile: path.path)
    }

    // MARK: - Delete Image

    func deleteImage(at filename: String) {
        let path = imagesDirectory.appendingPathComponent(filename)
        try? fileManager.removeItem(at: path)
    }

    // MARK: - Storage Info

    func totalStorageUsed() -> Int64 {
        var total: Int64 = 0
        if let files = try? fileManager.contentsOfDirectory(atPath: imagesDirectory.path) {
            for file in files {
                let path = imagesDirectory.appendingPathComponent(file).path
                if let attrs = try? fileManager.attributesOfItem(atPath: path),
                   let size = attrs[.size] as? Int64 {
                    total += size
                }
            }
        }
        return total
    }
}
