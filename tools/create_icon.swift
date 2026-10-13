// Rebuild the app icon using only macOS AppKit: swift tools/create_icon.swift assets
import AppKit

let output = URL(fileURLWithPath: CommandLine.arguments[1])
let iconset = output.appendingPathComponent("AyahStudio.iconset")
try FileManager.default.createDirectory(at: iconset, withIntermediateDirectories: true)
for size in [16, 32, 128, 256, 512] {
    for scale in [1, 2] {
        let pixels = size * scale
        let bitmap = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: pixels,
                                      pixelsHigh: pixels, bitsPerSample: 8,
                                      samplesPerPixel: 4, hasAlpha: true,
                                      isPlanar: false, colorSpaceName: .deviceRGB,
                                      bytesPerRow: 0, bitsPerPixel: 0)!
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: bitmap)
        let factor = CGFloat(pixels) / 128
        let transform = NSAffineTransform()
        transform.scale(by: factor)
        transform.concat()
        let tile = NSBezierPath(roundedRect: NSRect(x: 6, y: 6, width: 116, height: 116), xRadius: 27, yRadius: 27)
        NSGradient(starting: NSColor(srgbRed: 34/255, green: 38/255, blue: 43/255, alpha: 1),
                   ending: NSColor(srgbRed: 48/255, green: 52/255, blue: 58/255, alpha: 1))!
            .draw(in: tile, angle: 90)
        NSColor(srgbRed: 232/255, green: 197/255, blue: 114/255, alpha: 1).setStroke()
        for (x, height) in [(28, 12), (37, 26), (46, 40), (55, 56), (64, 34), (73, 22), (82, 40), (91, 26), (100, 12)] {
            let line = NSBezierPath()
            line.lineWidth = 7
            line.lineCapStyle = .round
            line.move(to: NSPoint(x: x, y: 64 - height/2))
            line.line(to: NSPoint(x: x, y: 64 + height/2))
            line.stroke()
        }
        NSColor(srgbRed: 250/255, green: 247/255, blue: 230/255, alpha: 1).setStroke()
        let cursor = NSBezierPath()
        cursor.lineWidth = 2.5
        cursor.lineCapStyle = .round
        cursor.move(to: NSPoint(x: 69, y: 29))
        cursor.line(to: NSPoint(x: 69, y: 97))
        cursor.stroke()
        NSColor(srgbRed: 250/255, green: 247/255, blue: 230/255, alpha: 1).setFill()
        NSBezierPath(ovalIn: NSRect(x: 66, y: 94, width: 6, height: 6)).fill()
        NSGraphicsContext.restoreGraphicsState()
        let suffix = scale == 2 ? "@2x" : ""
        try bitmap.representation(using: .png, properties: [:])!.write(
            to: iconset.appendingPathComponent("icon_\(size)x\(size)\(suffix).png"))
        if pixels == 1024 {
            try bitmap.representation(using: .png, properties: [:])!.write(
                to: output.appendingPathComponent("AyahStudio.png"))
        }
    }
}
let process = Process()
process.executableURL = URL(fileURLWithPath: "/usr/bin/iconutil")
process.arguments = ["-c", "icns", iconset.path, "-o", output.appendingPathComponent("AyahStudio.icns").path]
try process.run()
process.waitUntilExit()
guard process.terminationStatus == 0 else { exit(process.terminationStatus) }
try FileManager.default.removeItem(at: iconset)
