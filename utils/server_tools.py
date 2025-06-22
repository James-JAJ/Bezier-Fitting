# server_tools.py
# custom_print: 自定義的 print 函數，將輸出內容儲存到指定的全局變數中

# 全局變數，用於存儲 console_output 的引用
_console_output_ref = None

def set_console_output_ref(ref):
    """ 設置 console_output 的引用，應該在 app.py 初始化時調用
    Args:
        ref: console_output 的引用物件
        Datatype: 通常是 list 或其他可變物件的引用
    Returns:
        None
    ⚠️ 備註:
    - 必須在使用 custom_print 之前呼叫此函數
    - 用於建立全域引用，讓 custom_print 能存取輸出緩衝區
    """
    global _console_output_ref
    _console_output_ref = ref

def custom_print( *args, **kwargs):
    """ 自定義的 print 函數，可選擇直接輸出或儲存到緩衝區
    Args:
        ifsever (int): 控制輸出行為的旗標
            - 0: 直接印出到終端，不儲存
            - 1: 儲存到緩衝區，不顯示
        *args: 要印出的內容，支援多個參數
        **kwargs: print 函數的其他參數（目前未使用）
    Returns:
        None
    ⚠️ 備註:
    - ifsever=1 時需要先呼叫 set_console_output_ref 設定引用
    - 所有參數會被轉換為字串後用空格連接
    - 儲存模式下會自動在訊息後加上換行符號
    """
    global _console_output_ref
    message = " ".join(map(str, args))  # 將所有參數轉為字串

    if _console_output_ref is not None:
            _console_output_ref[0] += message + "\n"
    else:
        print("警告: console_output 引用尚未設置，請在 app.py 中調用 set_console_output_ref")
        