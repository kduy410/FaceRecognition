# FaceRecognition
Xây dựng hệ thống kiểm soát nhận dạng khuôn mặt với OpenCV Dlib và Deep Learning
https://techtalk.vn/xay-dung-he-thong-kiem-soat-nhan-dang-khuon-mat-voi-opencv-dlib-va-deep-learning.html?fbclid=IwAR1__VGmxhNzr59abtIN3g6WbD659sXNvuVB8zTQ2-G9SIizsbc3gx_zVjI

https://forum.machinelearningcoban.com/t/face-recognition-voi-keras-dlib-va-opencv/4688
Mục tiêu
Hiện nay có rất nhiều kỹ thuật để thực hiện việc nhận dạng khuôn mặt, tuy nhiên điểm chung của các kỹ thuật này là đều sẽ phải thực hiện qua 3 bước:
  1.Xác định và lấy ra (các) khuôn mặt có trong hình ảnh
  2.Từ hình ảnh các khuôn mặt lấy ra từ bước 1, thực hiện việc phân tích, trích xuất các đặt trưng của khuôn mặt
  3.Từ các thông tin có được sau khi phân tích, kết luận và xác minh danh tính người dùng
Thông qua bài viết lần này, mình sẽ xây dựng một hệ thống hoàn chỉnh cho việc nhận dạng khuôn mặt dựa vào thư viện Dlib của OpenCV và mạng Deep Learning sử dụng hàm Triplet Loss. Hy vọng sẽ giúp các bạn nắm được công nghệ này để có thể tự triển khai được trong thực tế.

Xác định khuôn mặt trong ảnh (Facial detection)
