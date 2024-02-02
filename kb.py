from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from datetime import datetime
import os, openai, json, csv, fitz

CUR_DATETIME = datetime.now()
CUR_DATE_TEXT = f"As of {CUR_DATETIME.strftime('%B %d, %Y')} ({CUR_DATETIME.strftime('%d/%m/%Y')}), "

CSV_PATH = './questions.csv'
EXTRACT_QUESTION_PER_CHARS = 80
EXTRACT_QUESTION_LIMIT = 15
QUESTION_MIN_CHARS = 10
ANSWER_MIN_CHARS = 10
INTRODUCTION = "You represent the AI-EP team. AI-EP is a project that leverages generative AI to provide parents that can't understand their child's Individualized Education Plan a translation, summary and interactive chatbot via a web application."
INSTRUCTIONS = "Please keep your tone polite and professional. Make sure the answer is structured, on point and concise. Avoid giving any information that is not relevant to the question."
TEMPLATE = INTRODUCTION + """Please answer this question:{question}\nTo assist in your response, here is the relevant data:{relevant_data}""" + INSTRUCTIONS
SYSTEM_PROMPT = "You are given some text. Respond in json format a list of {question_num} most important questions and a list of their answers labelled as 'questions' and 'answers' specfically. These question must be related to the IEP process, the AIEP project/team or its stakeholders."
PRIORITY_ANSWER = 'This information is verified and overrides others. {answer}'

class KnowledgeBase:
    def __init__(self, api_key=None, template=TEMPLATE, csv_path=CSV_PATH) -> None:
        openai.api_key = api_key if api_key else os.getenv("OPENAI_API_KEY")
        self.template, self.csv_path = template, csv_path
        self._load_embeddings()

    def _load_embeddings(self):
        loader = CSVLoader(file_path=self.csv_path)
        documents = loader.load()
        embeddings = OpenAIEmbeddings()
        self.embeddings_db = FAISS.from_documents(documents, embeddings)
    
    def _log(self, text):
        print(text)

    def update_from_pdf(self, file_name, file_byte_stream):
        doc = fitz.open(stream=file_byte_stream, filetype='pdf')
        page_count = doc.page_count
        for page_number in range(page_count):
            self._extract_data_from_page(file_name, doc[page_number], page_number+1, page_count+1)
        self._log('Embeddings Model Updated with PDF Data')
        
    def _extract_data_from_page(self, file_name, page, cur_page_num, total_page_num):
        self._log(f"Extracting Page {cur_page_num}/{total_page_num}")
        page_text = page.get_text()
        self._extract_data_from_text(page_text, file_name)
    
    def _extract_data_from_text(self, text, topic=None):
        char_num = len(text.strip())
        question_num = min(int(char_num / EXTRACT_QUESTION_PER_CHARS), EXTRACT_QUESTION_LIMIT)
        self._log(f"Generating {question_num} Question(s)")
        completion = openai.chat.completions.create(
            model='gpt-4-1106-preview',
            response_format={'type': 'json_object'},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(question_num=question_num)},
                {"role": "user", "content": f"Text: {text}"+ f"\nTopic/Source: {topic}" if topic else ''}
        ])
        data = json.loads(completion.choices[0].message.content)
        questions, answers = data.get('questions'), data.get('answers')
        self._log(f"{len(questions)} Questions and {len(answers)} Answers Generated")
        for i in range(question_num):
            question, answer = questions[i], answers[i]
            self.update_embeddings(question, answer)

    def update_embeddings(self, question, answer, priority=False):
        question_char_len, answer_char_len = len(question.strip()), len(answer.strip())
        if answer_char_len < ANSWER_MIN_CHARS: 
            self._log('Answer Data Too Short, Skipping Entry...')
            return
        elif question_char_len < QUESTION_MIN_CHARS: 
            if priority: 
                self._log('Insufficient Question Data, Extracting Question Based on Answers')
                self._extract_data_from_text(answer)
            else: 
                self._log('Generated Question Data Too Short, Skipping Entry...')
                return
        if priority:
            self._log('Updating Embeddings Model with High Priority Data')
            answer = PRIORITY_ANSWER.format(answer=answer) 
        with open(self.csv_path, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([CUR_DATE_TEXT + question, CUR_DATE_TEXT + answer])
        self._load_embeddings()
        if priority: self._log('Embeddings Model Updated')

    def _retrieve_info(self, query):
        similar_response = self.embeddings_db.similarity_search(query, k=3)
        page_contents_array = [doc.page_content for doc in similar_response]
        return page_contents_array
    
    def generate_response(self, question):
        relevant_data = self._retrieve_info(question)
        llm = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")
        prompt = PromptTemplate(
            input_variables=["question", "relevant_data"],
            template=self.template.format(question=CUR_DATE_TEXT + question,relevant_data=relevant_data)
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        response = chain.run(question=question, relevant_data=relevant_data)
        return response