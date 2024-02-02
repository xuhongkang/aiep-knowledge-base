from kb import KnowledgeBase
import streamlit as st
import fitz

class StreamlitKB(KnowledgeBase):
    def _log(self, text):
        st.write(text)
        return super()._log(text)

    def update_from_pdf(self, file_name, file_byte_stream, st_progress_func):
        doc = fitz.open(stream=file_byte_stream, filetype='pdf')
        page_count = doc.page_count
        for page_number in range(page_count):
            self._extract_data_from_page(file_name, doc[page_number], page_number+1, page_count+1, st_progress_func)
        self._log('PDF data added')
    
    def _extract_data_from_page(self, file_name, page, cur_page_num, total_page_num, st_progress_func):
        cur_progress = int((cur_page_num - 1)/total_page_num * 100)
        st_progress_func(cur_progress)
        return super()._extract_data_from_page(file_name, page, cur_page_num, total_page_num)