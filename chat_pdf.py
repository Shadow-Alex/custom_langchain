def process_pdf(device, q):
    import json, os
    from langchain.document_loaders import Pix2TextLoader
    print("Start ocr process on device : " + str(device))
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    loader = Pix2TextLoader("./1685435898.9404118_herd-scharfstein.pdf", device=device)
    while True:
        loader.file_path = q.get() # 接收新的pdf路径
        if loader.file_path is None:
            return
        import time
        start = time.time()
        documents = loader.load()
        end = time.time()
        with open(os.path.join("ocr_results", os.path.split(loader.file_path)[1] + '.json'), 'w') as f:
            s = ''
            for doc in documents:
                s += doc.page_content
                s += '\n\n'
            json.dump(s, f)
        print(loader.file_path + " used " + str(end - start) + " secs.")

def parse_command():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--on", type=lambda s: [str(int(item)) for item in s.split(',')])
    args = parser.parse_args()
    return args.on

if __name__ == "__main__":
    import os
    from multiprocessing import Process, Queue
    from itertools import cycle

    pdf_directory = "/pdfs/"
    pdf_files = [os.path.join(pdf_directory, f) for f in os.listdir(pdf_directory) if f.endswith('.pdf')]
    pdf_files = [f for f in pdf_files if not os.path.exists(os.path.join("ocr_results", os.path.split(f)[1] + '.json'))]

    devices = parse_command()
    procs = len(devices)

    import time
    start = time.time()
    q_list = []
    workers = []
    for i in range(procs):
        q = Queue()
        q_list.append(q)
        worker = Process(target=process_pdf, args=(devices.pop(), q))
        workers.append(worker)
        worker.start()
    # tasks.
    for file_path, queue in zip(pdf_files, cycle(q_list)):
        queue.put(file_path)
    
    # close.
    for queue in q_list:
        queue.put(None)
    
    for worker in workers:
        worker.join()
    
    end = time.time()
    print("Total : " + str(end - start) + " secs.")

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size = 4000,
#     chunk_overlap  = 200,
#     length_function = len,
#     keep_separator = False,
#     separators = ['\n\n', "\n", " ", ""]
# )
#
# texts = text_splitter.split_documents(documents)
#
# print("Hi.")
