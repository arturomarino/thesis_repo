"""Replace Italian figure annotations while preserving every plotted data pixel."""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "img"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"


def font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT, size)


def centered(draw: ImageDraw.ImageDraw, box, text: str, text_font, fill="black"):
    left, top, right, bottom = box
    bounds = draw.textbbox((0, 0), text, font=text_font)
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    draw.text(
        ((left + right - width) / 2, (top + bottom - height) / 2 - bounds[1]),
        text,
        font=text_font,
        fill=fill,
    )


def rotated_label(image: Image.Image, box, text: str, size: int) -> None:
    left, top, right, bottom = box
    layer = Image.new("RGBA", (bottom - top, right - left), (255, 255, 255, 0))
    draw = ImageDraw.Draw(layer)
    centered(draw, (0, 0, layer.width, layer.height), text, font(size))
    layer = layer.rotate(90, expand=True)
    image.alpha_composite(layer, (left, top))


def translate_learning_curve() -> None:
    image = Image.open(IMG / "learning-curve.png").convert("RGBA")
    draw = ImageDraw.Draw(image)

    draw.rectangle((0, 0, 1581, 58), fill="white")
    centered(
        draw,
        (0, 0, 1581, 55),
        "Learning curves and persistence comparison",
        font(28),
    )

    # Replace the lower-panel y label and x label only; data axes are untouched.
    draw.rectangle((0, 820, 52, 1195), fill="white")
    rotated_label(image, (3, 852, 48, 1168), "Normalized RMSE", 22)
    draw.rectangle((720, 1365, 880, 1413), fill="white")
    centered(draw, (720, 1365, 880, 1410), "Epoch", font(22))

    # Rebuild the lower legend with the same colours and symbols.
    box = (1153, 741, 1553, 941)
    draw.rounded_rectangle(box, radius=4, fill=(255, 255, 255, 242), outline=(200, 200, 200), width=2)
    rows = [
        (770, "Training RMSE (fixed weights)", "line-blue"),
        (807, "Validation RMSE", "line-red"),
        (844, "Training persistence RMSE", "dot-blue"),
        (881, "Validation persistence RMSE", "dash-purple"),
    ]
    for y, label, kind in rows:
        if kind == "line-blue":
            draw.line((1168, y, 1213, y), fill=(40, 103, 224), width=4)
            draw.ellipse((1187, y - 4, 1195, y + 4), fill=(40, 103, 224))
        elif kind == "line-red":
            draw.line((1168, y, 1213, y), fill=(220, 45, 40), width=4)
            draw.ellipse((1187, y - 4, 1195, y + 4), fill=(220, 45, 40))
        elif kind == "dot-blue":
            for x in range(1168, 1213, 9):
                draw.line((x, y, x + 3, y), fill=(40, 103, 224), width=3)
        else:
            for x in range(1168, 1213, 14):
                draw.line((x, y, min(x + 8, 1213), y), fill=(121, 58, 238), width=3)
        draw.text((1223, y - 12), label, font=font(18), fill="black")

    # Best-epoch marker sits below the four line entries.
    y = 918
    draw.ellipse((1182, y - 8, 1198, y + 8), fill=(15, 166, 88))
    draw.text((1223, y - 12), "Best validation (epoch 94)", font=font(18), fill="black")

    image.convert("RGB").save(IMG / "learning-curve-english.png", quality=95)


def translate_map(source: str, target: str, title: str, subtitle: str, colorbar: str) -> None:
    image = Image.open(IMG / source).convert("RGBA")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    draw.rectangle((0, 0, width - 285, 160), fill="white")
    centered(draw, (0, 43, width - 285, 81), title, font(27))
    centered(draw, (0, 80, width - 285, 118), subtitle, font(24))

    draw.rectangle((885, 1310, 1530, 1375), fill="white")
    centered(draw, (885, 1310, 1530, 1360), "Longitude (°)", font(24))

    draw.rectangle((0, 450, 39, 975), fill="white")
    rotated_label(image, (2, 515, 37, 925), "Latitude (°)", 24)

    draw.rectangle((2490, 260, width, 1150), fill="white")
    rotated_label(image, (2500, 370, width - 8, 1040), colorbar, 23)

    image.convert("RGB").save(IMG / target, quality=95)


if __name__ == "__main__":
    translate_learning_curve()
    translate_map(
        "temperature-forecast-1999-08-15.png",
        "temperature-forecast-1999-08-15-english.png",
        "Sea-temperature forecast — 15 August 1999",
        "Input ending 14 August → t+1 | depth 0.51 m | epoch-94 checkpoint | mean μ",
        "Forecast temperature (°C)",
    )
    translate_map(
        "temperature-error-1999-08-15.png",
        "temperature-error-1999-08-15-english.png",
        "Sea-temperature forecast error — 15 August 1999",
        "forecast − target | input ending 14 August → t+1 | depth 0.51 m | epoch-94 checkpoint",
        "Error: forecast − target (°C)",
    )
