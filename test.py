from utils import load_model, extend_maps, prepocess_data_for_lstmcrf
from loader import build_corpus
from evaluate import Metrics

# 训练好的模型的存放路径
CRF_MODEL_PATH = 'model/crf.pkl'
BiLSTMCRF_MODEL_PATH = 'model/bilstm_crf.pkl'


def main():
    print("测试集评估结果...")

    print("读取数据...")
    train_word_lists, train_tag_lists, word2id, tag2id = build_corpus("train")
    test_word_lists, test_tag_lists = build_corpus("test", make_vocab=False)


    print("CRF...")
    crf_model = load_model(CRF_MODEL_PATH)
    crf_pred = crf_model.test(test_word_lists)
    metrics = Metrics(test_tag_lists, crf_pred, remove_O=False)
    metrics.report_scores()
    #metrics.report_confusion_matrix()


    print("BILSTM+CRF...")
    crf_word2id, crf_tag2id = extend_maps(word2id, tag2id)
    bilstm_model = load_model(BiLSTMCRF_MODEL_PATH)
    # bilstm_model.model.bilstm.bilstm.flatten_parameters()  # remove warning when CUDA
    test_word_lists, test_tag_lists = prepocess_data_for_lstmcrf(
        test_word_lists, test_tag_lists, test=True
    )
    lstmcrf_pred, target_tag_list = bilstm_model.test(test_word_lists, test_tag_lists,
                                                      crf_word2id, crf_tag2id)
    metrics = Metrics(target_tag_list, lstmcrf_pred, remove_O=False)
    metrics.report_scores()
    #metrics.report_confusion_matrix()



if __name__ == "__main__":
    main()
