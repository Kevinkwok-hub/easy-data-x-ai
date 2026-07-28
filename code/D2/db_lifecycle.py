def close_database(database):
    """兼容 close() 和上下文管理协议两种 pyseekdb 客户端。"""
    close = getattr(database, "close", None)
    if callable(close):
        close()
        return
    exit_method = getattr(database, "__exit__", None)
    if callable(exit_method):
        exit_method(None, None, None)
