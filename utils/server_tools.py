# server_tools.py
# custom_print: 自定義的 print 函數，將輸出內容儲存到指定的全局變數中

# 全局變數，用於存儲 console_output 的引用
_console_output_ref = None

def set_console_output_ref(ref):
    """設置 console_output 的引用，應該在 app.py 初始化時調用"""
    global _console_output_ref
    _console_output_ref = ref

def custom_print(ifsever=1,*args, **kwargs):
    """自定義的 print 函數，將輸出內容儲存到 console_output 變數中"""
    global _console_output_ref
    if ifsever == 1:
        message = " ".join(map(str, args))  # 將所有參數轉換為字串並連接起來
        print(message)
        
        # 確認 _console_output_ref 已設置
        if _console_output_ref is not None:
            _console_output_ref[0] += str(message) + "\n"  # 將訊息加入到 console_output 字串中，並加上換行符號
        else:
            print("警告: console_output 引用尚未設置，請在 app.py 中調用 set_console_output_ref")
    else:
        print(message)
