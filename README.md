VERSION USES PYTHON 3.7

# FaceRecognition
<br>Xây dựng hệ thống kiểm soát nhận dạng khuôn mặt với OpenCV Dlib và Deep Learning</br>
https://techtalk.vn/xay-dung-he-thong-kiem-soat-nhan-dang-khuon-mat-voi-opencv-dlib-va-deep-learning.html?fbclid=IwAR1__VGmxhNzr59abtIN3g6WbD659sXNvuVB8zTQ2-G9SIizsbc3gx_zVjI

https://forum.machinelearningcoban.com/t/face-recognition-voi-keras-dlib-va-opencv/4688
Mục tiêu
Hiện nay có rất nhiều kỹ thuật để thực hiện việc nhận dạng khuôn mặt, tuy nhiên điểm chung của các kỹ thuật này là đều sẽ phải thực hiện qua 3 bước:
  1.Xác định và lấy ra (các) khuôn mặt có trong hình ảnh
  2.Từ hình ảnh các khuôn mặt lấy ra từ bước 1, thực hiện việc phân tích, trích xuất các đặt trưng của khuôn mặt
  3.Từ các thông tin có được sau khi phân tích, kết luận và xác minh danh tính người dùng
Thông qua bài viết lần này, mình sẽ xây dựng một hệ thống hoàn chỉnh cho việc nhận dạng khuôn mặt dựa vào thư viện Dlib của OpenCV và mạng Deep Learning sử dụng hàm Triplet Loss. Hy vọng sẽ giúp các bạn nắm được công nghệ này để có thể tự triển khai được trong thực tế.

Xác định khuôn mặt trong ảnh (Facial detection)

AIviVN Celebs Re-identification Baseline
Phương pháp chung
Rất đơn giản thôi, mình sử dụng pretrained facenet từ repo này https://github.com/nyoki-mtl/keras-facenet . Tính embedding cho mỗi ảnh, so sánh mỗi embedding mỗi ảnh trong tập test với từng nhóm các embeddings thuộc cùng một người trong tập train, tìm ra khoảng cách ngắn nhất của mỗi ảnh đến mỗi người
