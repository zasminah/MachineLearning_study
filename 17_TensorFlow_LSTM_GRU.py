import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM, GRU, Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd

tf.random.set_seed(1)
np.random.seed(1)

paragraph_dict_list = [
    {
        'paragraph': 'dishplace is located in sunnyvale downtown there is parking around the area but it can be difficult to find during peak business hours my sisters and i came to this place for dinner on a weekday they were really busy so i highly recommended making reservations unless you have the patience to wait',
        'category': 'food'},
    {
        'paragraph': 'service can be slower during busy hours but our waiter was courteous and help gave some great entree recommendations',
        'category': 'food'},
    {
        'paragraph': 'portions are huge both french toast and their various omelettes are really good their french toast is probably 1.5x more than other brunch places great place to visit if you are hungry and dont want to wait 1 hour for a table',
        'category': 'food'},
    {
        'paragraph': 'we started with apps going the chicken and waffle slides and chicken nachos the sliders were amazing and the nachos were good too maybe by themselves the nachos would have scored better but after those sliders they were up against some tough competition',
        'category': 'food'},
    {
        'paragraph': 'the biscuits and gravy was too salty two people in my group had the gravy and all thought it was too salty my hubby ordered a side of double egg and it was served on two small plates who serves eggs to one person on separate plates we commented on that when it was delivered and even the server laughed and said she doesnt know why the kitchen does that presentation of food is important and they really missed on this one',
        'category': 'food'},
    {
        'paragraph': 'the garlic fries were a great starter (and a happy hour special) the pancakes looked and tasted great and were a fairly generous portion',
        'category': 'food'},
    {
        'paragraph': 'our meal was excellent i had the pasta ai formaggi which was so rich i didnt dare eat it all although i certainly wanted to excellent flavors with a great texture contrast between the soft pasta and the crisp bread crumbs too much sauce for me but a wonderful dish',
        'category': 'food'},
    {
        'paragraph': 'what i enjoy most about palo alto is so many restaurants have dog-friendly seating outside i had bookmarked italico from when they first opened about a 1.5 years ago and was jonesing for some pasta so time to finally knock that bookmark off',
        'category': 'food'},
    {
        'paragraph': 'the drinks came out fairly quickly a good two to three minutes after the orders were taken i expected my iced tea to taste a bit more sweet but this was straight up green tea with ice in it not to complain of course but i was pleasantly surprised',
        'category': 'food'},
    {
        'paragraph': 'despite the not so good burger the service was so slow the restaurant wasnt even half full and they took very long from the moment we got seated to the time we left it was almost 2 hours we thought that it would be quick since we ordered as soon as we sat down my coworkers did seem to enjoy their beef burgers for those who eat beef however i will not be returning it is too expensive and extremely slow service',
        'category': 'food'},

    {
        'paragraph': 'the four reigning major champions simona halep caroline wozniacki angelique kerber and defending us open champion sloane stephens could make a case for being the quartet most likely to succeed especially as all but stephens has also enjoyed the no1 ranking within the last 14 months as they prepare for their gruelling new york campaigns they currently hold the top four places in the ranks',
        'category': 'sports'},
    {
        'paragraph': 'the briton was seeded nn7 here last year before a slump in form and confidence took her down to no46 after five first-round losses but there have been signs of a turnaround including a victory over a sub-par serena williams in san jose plus wins against jelena ostapenko and victoria azarenka in montreal. konta pulled out of new haven this week with illness but will hope for good things where she first scored wins in a major before her big breakthroughs to the semis in australia and wimbledon',
        'category': 'sports'},
    {
        'paragraph': 'stephens surged her way back from injury in stunning style to win her first major here last year—and ranked just no83 she has since proved what a big time player she is winning the miami title via four fellow major champions then reaching the final at the french open back on north american hard courts she ran to the final in montreal only just edged out by halep she has also avoided many of the big names in her quarter—except for wild card azarenka as a possible in the third round',
        'category': 'sports'},
    {
        'paragraph': 'when it came to england chances in the world cup it would be fair to say that most fans had never been more pessimistic than they were this year after enduring years of truly dismal performances at major tournaments – culminating in the 2014 event where they failed to win any of their three group games and finished in bottom spot those results led to the resignation of manager roy hodgson',
        'category': 'sports'},
    {
        'paragraph': 'the team that eliminated russia – croatia – also improved enormously during the tournament before it began their odds were 33/1 but they played with real flair and star players like luka modric ivan rakitic and ivan perisic showed their quality on the world stage having displayed their potential by winning all three of their group stage games croatia went on to face difficult tests like the semi-final against england',
        'category': 'sports'},
    {
        'paragraph': 'the perseyside outfit finished in fourth place in the premier league table and without a trophy last term after having reached the champions league final before losing to real madrid',
        'category': 'sports'},
    {
        'paragraph': 'liverpool fc will return to premier league action on saturday lunchtime when they travel to leicester city in the top flight as they look to make it four wins in a row in the league',
        'category': 'sports'},
    {
        'paragraph': 'alisson signed for liverpool fc from as roma this summer and the brazilian goalkeeper has helped the reds to keep three clean sheets in their first three premier league games',
        'category': 'sports'},
    {
        'paragraph': 'but the rankings during that run-in to new york hid some very different undercurrents for murray had struggled with a hip injury since the clay swing and had not played a match since losing his quarter-final at wimbledon and he would pull out of the us open just two days before the tournament began—too late however to promote nederer to the no2 seeding',
        'category': 'sports'},
    {
        'paragraph': 'then came the oh-so-familiar djokovic-nadal no-quarter-given battle for dominance in the third set there were exhilarating rallies with both chasing to the net both retrieving what looked like winning shots nadal more than once pulled off a reverse smash and had his chance to seal the tie-break but it was djokovic serving at 10-9 who dragged one decisive error from nadal for a two-sets lead',
        'category': 'sports'}
]
df = pd.DataFrame(paragraph_dict_list)
df = df[['paragraph', 'category']]
print(df.head())

def get_vocab_size(df): # 최대 들어간 지문 크기, 단어들을 one-hot code, set 이용 unique한 값 뽑기
    results = set()
    df['paragraph'].str.lower().str.split().apply(results.update)
    return len(results)

vocab_size = get_vocab_size(df)
print('vocab_size:', vocab_size)

paragraphs = df['paragraph'].tolist()
encoded_paragraphs = [one_hot(paragraph, vocab_size) for paragraph in paragraphs]


def get_max_length(df): #가장 긴 지문 길이
    max_length = 0
    for row in df['paragraph']:
        if len(row.split(" ")) > max_length:
            max_length = len(row.split(" "))
    return max_length


max_length = get_max_length(df)
print(max_length)

padded_paragraphs_encoding = pad_sequences(encoded_paragraphs, maxlen=max_length, padding='post') #지문길이 맞춰줌 최대길이로..+add padding post는 padding을 뒤에 넣겠다
print(padded_paragraphs_encoding)

categories = df['category'].tolist()
def category_encode(category):
    if category == 'food':
        return [1,0]
    else:
        return [0,1]
encoded_category = [category_encode(category) for category in categories]

model = Sequential()
model.add(Embedding(vocab_size, 5, input_length=max_length)) #Embedding
model.add(LSTM(64))
model.add(Dense(32, activation='relu')) #FC Layer 사용
model.add(Dense(2, activation='softmax')) #결과값을 softmax 사용용
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()
x_train = np.array(padded_paragraphs_encoding)
y_train = np.array(encoded_category)
model.fit(x_train, y_train,batch_size=10,epochs=50)