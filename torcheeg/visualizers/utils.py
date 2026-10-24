import io

from PIL import Image


def _plot_to_image(ploter):
    buf = io.BytesIO()
    ploter.savefig(buf, bbox_inches='tight', pad_inches=0.0)
    buf.seek(0)
    return Image.open(buf)

