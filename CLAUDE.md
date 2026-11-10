# search-engine: quy định riêng của repo

Đặc tả đầy đủ nằm ở `SPEC.md`. Quy trình giao việc cho Gemini ở `~/.claude/CLAUDE.md` áp dụng, với các điều chỉnh sau (ưu tiên hơn file global):

- Sau khi người dùng duyệt một task, Claude được phép commit ở repo chính (không push), rồi mới tạo worktree cho task kế tiếp. Các task chạy tuần tự vì task sau dùng code của task trước.
- `data/` không nằm trong git. Mỗi worktree tạo junction `data` trỏ về `data/` của repo chính để Gemini đọc `data/raw` và ghi `data/processed`.
- Trước `git worktree remove`, phải gỡ junction bằng `cmd //c rmdir <worktree>\data` (chỉ gỡ liên kết, không xoá dữ liệu).
- Máy có Python 3.10 (spec ghi 3.11): code phải chạy được trên 3.10. GPU RTX 3050 chỉ có 4 GB VRAM.
- Claude không tự viết code, không tự test. Cần kiểm tra thì ghi lệnh/input vào spec cho Gemini chạy, rồi đọc output Gemini ghi ra file.
