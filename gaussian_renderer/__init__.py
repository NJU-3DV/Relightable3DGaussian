from gaussian_renderer.render import render
from gaussian_renderer.neilf import render_neilf


render_fn_dict = {
    "render": render,
    "neilf": render_neilf,
}