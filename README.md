# modeltuner

## 1. Setup dataset:
   a. Chỉnh path đến csv trong main.py
   b. Chỉnh path đến image/ trong config.py

TO NOTE: chưa có thì tải về bằng data_downloader.py => Nhớ vô đó setup lại path cho đúng.

## 2. Config:
   a. batch_size: tùy theo vram của GPU mà set: <8GB thì set 4, <16GB thì set 8, <32GB thì set 16, còn lại thì cứ chơi 32

   b. stage1_epochs và stage2_epochs là gate để bặt tắt mở thêm layer cho Vit, BART,...

   c. unfreeze_last_n_layers mở layer cuối thì mở bao nhiêu layers? ( mặc định để 4 )

   d. evaluate_every_n_steps: tiến hành debug xem mô hình có học ổn hay không?

   e. num_epochs: thường thì từ 10-15, nhưng mà tầm 10 là ổn rồi.

## 3. Các model khác có thể thử:

### a. Vision Model (`vision_model`)

Mục tiêu là sử dụng các model có khả năng trích xuất đặc trưng hình ảnh phong phú và chi tiết hơn.

*   **Nâng cấp trực tiếp (cùng kiến trúc ViT):**
    *   `google/vit-large-patch16-224-in21k`: Phiên bản "large" của ViT-Base, có nhiều tham số hơn và khả năng học được các đặc trưng phức tạp hơn.
*   **Model được pre-train với phương pháp tốt hơn (BEiT/MAE):**
    *   `microsoft/beit-large-patch16-224-in22k`: **(Đề xuất cao)**. BEiT (Bidirectional Encoder representation from Image Transformers) là một trong những phương pháp pre-train hiệu quả nhất cho vision. Tên dự án của bạn là "BARTphoBEIT", nên việc sử dụng BEiT là rất phù hợp.
    *   `facebook/vit-mae-large-patch16-224`: Pre-train bằng Masked Autoencoders, một phương pháp self-supervised rất mạnh mẽ.
*   **Model được pre-train trên dữ liệu Image-Text (CLIP):**
    *   `openai/clip-vit-large-patch14`: **(Đề xuất cao)**. CLIP được huấn luyện trên hàng trăm triệu cặp (ảnh, mô tả), giúp nó học được các đặc trưng hình ảnh có sự liên kết chặt chẽ với ngữ nghĩa ngôn ngữ. Đây là lựa chọn rất mạnh cho các bài toán đa phương thức như VQA.
*   **Sử dụng độ phân giải cao hơn:**
    *   `google/vit-base-patch16-384`: Sử dụng ảnh đầu vào 384x384 thay vì 224x224 có thể giúp model nhận diện các vật thể hoặc chi tiết nhỏ tốt hơn.

### b. Text Encoder (`text_model`)

`vinai/phobert-large` đã là một lựa chọn rất mạnh. Tuy nhiên, bạn có thể thử các phiên bản mới hơn hoặc các model khác.

*   **Phiên bản mới của PhoBERT:**
    *   `vinai/phobert-base-v2`: VinAI đã phát hành phiên bản 2 của PhoBERT với cải tiến về pre-training. Đây là một lựa chọn đáng để thử nghiệm.
*   **Model BERT tiếng Việt khác:**
    *   `FPTAI/v-bert-base-cased`: Một model BERT khác dành cho tiếng Việt từ FPT AI.
*   **Model đa ngôn ngữ mạnh mẽ:**
    *   `xlm-roberta-large`: Mặc dù là model đa ngôn ngữ, XLM-RoBERTa-Large có hiệu năng rất cao trên nhiều ngôn ngữ, bao gồm cả tiếng Việt, và có thể mang lại góc nhìn khác so với PhoBERT.

### c. Decoder Model (`decoder_model`)

`vinai/bartpho-word` là một lựa chọn tốt. Các lựa chọn thay thế có thể là các model sinh văn bản (generative) lớn hơn.

*   **Phiên bản lớn hơn của BARTpho:**
    *   `vinai/bartpho-syllable`: Sử dụng tokenization theo âm tiết thay vì từ, có thể xử lý các từ OOV (out-of-vocabulary) tốt hơn.
*   **Model T5 cho tiếng Việt:**
    *   `VietAI/vit5-large`: ViT5 là phiên bản T5 được pre-train cho tiếng Việt. Kiến trúc Encoder-Decoder của T5 rất phù hợp cho các tác vụ sinh văn bản có điều kiện như VQA.
*   **Model GPT tiếng Việt (nếu muốn thử kiến trúc mới):**
    *   `Viet-Mistral/Vistral-7B-Chat`: Sử dụng một model GPT lớn làm decoder là một hướng đi phức tạp hơn nhưng có tiềm năng tạo ra câu trả lời tự nhiên và đa dạng hơn. Tuy nhiên, việc tích hợp sẽ đòi hỏi thay đổi kiến trúc đáng kể.

### TO NOTE: Lưu ý là mỗi model có cách load pretrain khác nhau, dim khác nhau, tokenizer cũng có thể khác nhau nên khi chỉnh sửa thì cần cẩn thận.