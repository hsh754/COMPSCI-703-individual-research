from src.datasets.loader import load_commonsenseqa, load_cose

if __name__ == "__main__":
    print("正在下载 CommonsenseQA ...")
    load_commonsenseqa()
    print("CommonsenseQA 下载完成！")

    print("正在下载 CoS-E ...")
    load_cose()
    print("CoS-E 下载完成！") 