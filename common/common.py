import streamlit as st
import sys
import importlib
import traceback
import os
from typing import Optional

# Hàm chạy một file Streamlit ổn định
def run_stable_script(script_path: str) -> None:
    st.title("🌟 Chạy Ứng dụng Streamlit Ổn Định")

    # Kiểm tra file tồn tại
    if not os.path.exists(script_path):
        st.error(f"File '{script_path}' không tồn tại. Vui lòng kiểm tra đường dẫn.")
        return

    # Hiển thị thông báo trạng thái
    st.subheader(f"Đang chạy file: {script_path}")
    st.write("Nếu có lỗi, ứng dụng sẽ không sập mà hiển thị thông báo chi tiết.")

    try:
        # Import và chạy file Streamlit
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        spec = importlib.util.spec_from_file_location(module_name, script_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Gọi hàm chính (nếu có) hoặc chạy trực tiếp
        if hasattr(module, "__main__"):
            module.__main__()
        elif hasattr(module, "main"):
            module.main()
        else:
            st.warning("Không tìm thấy hàm chính trong file. Vui lòng đảm bảo file có hàm `if __name__ == '__main__':`.")

    except Exception as e:
        st.error("Ứng dụng gặp lỗi, nhưng đã được xử lý để không sập:")
        st.write(f"**Lỗi chi tiết:** {str(e)}")
        st.write("**Traceback:**")
        st.code(traceback.format_exc(), language="python")
        st.write("Vui lòng kiểm tra file bài tập hoặc báo cáo lỗi để được hỗ trợ.")

    st.write("---")
    st.markdown("Cảm ơn bạn đã sử dụng ứng dụng ổn định! 🌟 Nếu có thắc mắc, hãy liên hệ với chúng tôi.")

# Hàm chính
if __name__ == "__main__":
    # Đường dẫn đến file bài tập bạn muốn chạy (ví dụ: train_clustering.py hoặc view_clustering_logs.py)
    script_path = st.text_input("Nhập đường dẫn đến file Streamlit (ví dụ: exercises/exercise_4/train_clustering.py)", 
                               value="exercises/exercise_4/train_clustering.py")

    if st.button("Chạy ứng dụng", key="run_button"):
        run_stable_script(script_path)
    
    # Nút làm mới để thử lại nếu có lỗi
    if st.button("Làm mới ứng dụng", key="refresh_button"):
        st.experimental_rerun()