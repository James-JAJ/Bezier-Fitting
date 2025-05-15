#custom_print:自定義的 print 函數，將輸出內容儲存到 console_output 變數中。
def custom_print(*args, **kwargs):
    global console_output
    message = " ".join(map(str, args))  # 將所有參數轉換為字串並連接起來
    print(message)
    console_output += message + "\n"  # 將訊息加入到 console_output 字串中，並加上換行符號
