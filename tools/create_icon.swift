// Rebuild the app icon using only macOS AppKit: swift tools/create_icon.swift assets
import AppKit

let output = URL(fileURLWithPath: CommandLine.arguments[1])
let iconset = output.appendingPathComponent("AyahStudio.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let image = NSImage(size: NSSize(width: pixels, height: pixels))
        image.lockFocus()
        let factor = CGFloat(pixels) / 128
        let transform = NSAffineTransform()
        transform.scale(by: factor)
        transform.concat()
        NSColor(srgbRed: 36/255, green: 59/255, blue: 57/255, alpha: 1).setFill()
        NSBezierPath(roundedRect: NSRect(x: 6, y: 6, width: 116, height: 116), xRadius: 27, yRadius: 27).fill()
        NSColor(srgbRed: 232/255, green: 197/255, blue: 114/255, alpha: 1).setStroke()
        for (x, height) in [(28, 12), (42, 40), (57, 64), (72, 26), (87, 46), (101, 12)] {
            let line = NSBezierPath()
            line.lineWidth = 7
            line.lineCapStyle = .round
            line.move(to: NSPoint(x: x, y: 64 - height/2))
            line.line(to: NSPoint(x: x, y: 64 + height/2))
            line.stroke()
        }
        image.unlockFocus()
        let bitmap = NSBitmapImageRep(data: image.tiffRepresentation!)!
        let suffix = scale == 2 ? "@2x" : ""
        try bitmap.representation(using: .png, properties: [:])!.write(
            to: iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
    }
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", output.appendingPathComponent("AyahStudio.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { exit(process.terminationStatus) }
try FileManager.default.removeItem(at: iconset)
