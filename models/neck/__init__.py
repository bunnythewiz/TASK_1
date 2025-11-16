from .FPN import FPN

__all__ = ['build_neck']

# Only support FPN (what you have)
support_neck = ['FPN']


def build_neck(neck_name, **kwargs):
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck