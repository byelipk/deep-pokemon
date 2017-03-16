def read_target_label(lower=False):
    f = open("target_label.txt", "r")
    l = f.read().strip()
    
    if lower == True:
        l = l.lower()

    f.close()
    return l
