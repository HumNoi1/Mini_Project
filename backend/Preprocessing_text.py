from string import capwords
import pandas as pd
import re
import nltk
from pythainlp.tokenize import word_tokenize

# Step 1.1: Preprocess Tweets
def preprocess_tweet(tweet, language='en'):
    if tweet is None:
        return ""  # คืนค่าว่างถ้า tweet เป็น None

    tweet = str(tweet)  # ทำให้มั่นใจว่าเป็นสตริง

    if language == 'th':
        # สำหรับภาษาไทย
        tweet = word_tokenize(tweet)  # ใช้ pythainlp ในการตัดคำภาษาไทย

        # ลบตัวอักษรพิเศษ
        tweet = re.sub(r'[^ก-๙a-zA-Z0-9 ]+', '', ' '.join(tweet))  # แปลงเป็นสตริงก่อน
        tweet = tweet.lower()

        # โหลด stopwords ภาษาไทย (สามารถใช้จาก `stopwords` หรือสร้างเอง)
        stop_words_th = set([
            'แต่', 'และ', 'ที่', 'จาก', 'เป็น', 'จะ', 'กับ', 'ใน', 'ได้', 'ดังนั้น', 'จากนั้น', 'ซึ่ง', 'นั้น', 'หรือ'
        ])  # เพิ่มเติมคำ stopwords ของคุณเอง

        # ลบคำ stopwords ภาษาไทย
        words = tweet.split()
        words = [word for word in words if word not in stop_words_th]

    else:
        # สำหรับภาษาอังกฤษ
        nltk.download('stopwords')
        stop_words_en = set(capwords.words('english'))

        # ลบ mentions, hashtags, URLs, และตัวอักษรพิเศษ
        tweet = re.sub(r'@\w+|#\w+|http\S+|[^A-Za-z0-9 ]+', '', tweet)
        tweet = tweet.lower()

        # ลบ stopwords
        words = tweet.split()
        words = [word for word in words if word not in stop_words_en]

    return ' '.join(words)


# Step 2: Read Dataset and Apply Preprocessing
def main(file_path):
    # Read the dataset
    df = pd.read_excel(file_path, sheet_name='data')  # อ่านจาก sheet ชื่อ 'data'

    # ตรวจสอบว่าในคอลัมน์ 'tweetText' มีค่าเป็น None หรือไม่
    df['tweetText'] = df['tweetText'].fillna('')

    # ประมวลผลข้อมูล
    df['processed_tweet'] = df['tweetText'].apply(preprocess_tweet, language='th')

    # แสดงตัวอย่างข้อมูล 2 แถวจากแต่ละคลาส
    print("ตัวอย่างข้อมูลที่ผ่านการ preprocess:")
    for label in df['hashtage'].unique():
        print(f"\nคลาส: {label}")
        sample = df[df['hashtage'] == label].head(2)  # เลือก 2 แถวแรกจากแต่ละคลาส
        print(sample[['tweetText', 'processed_tweet']])

    # Save the processed data to a new file
    df.to_excel('data/new_data1.xlsx', index=False)
    print("Processed data has been saved to 'new1_data1.xlsx'")

# Run the main function with the path to your dataset
main('data/fortrain.xlsx')