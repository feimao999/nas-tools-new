from app.utils.commons import singleton


@singleton
class DisplayHelper(object):
    """
    轻量空壳实现：已移除 Xvfb / PyVirtualDisplay 虚拟桌面组件
    """
    _display = None

    def __init__(self):
        pass

    def init_config(self):
        pass

    def get_display(self):
        return None

    def stop_service(self):
        pass

    @staticmethod
    def can_display():
        return False
