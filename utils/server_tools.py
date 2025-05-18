# server_tools.py
# custom_print: 自定義的 print 函數，將輸出內容儲存到指定的全局變數中

# 全局變數，用於存儲 console_output 的引用
_console_output_ref = None

def set_console_output_ref(ref):
    """設置 console_output 的引用，應該在 app.py 初始化時調用"""
    global _console_output_ref
    _console_output_ref = ref

def custom_print(ifsever=1, *args, **kwargs): 
    """自定義的 print 函數
    
    參數:
        ifsever = 1 時：儲存訊息，不顯示
        ifsever = 0 時：直接印出，不儲存
    """
    global _console_output_ref
    message = " ".join(map(str, args))  # 將所有參數轉為字串

    if ifsever == 0:
        print(message)  # 直接輸出訊息
    elif ifsever == 1:
        if _console_output_ref is not None:
            _console_output_ref[0] += message + "\n"
        else:
            print("警告: console_output 引用尚未設置，請在 app.py 中調用 set_console_output_ref")

