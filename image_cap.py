# Thêm thư viện
import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt
import string
import os
from PIL import Image
import glob
from pickle import dump, load
from time import time
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras import Input, layers
from keras import optimizers
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

########################### I. PHẦN ĐỌC CÁC THÔNG TIN MÔ TẢ ##################################
path_to_desc = "Flickr8k/Flickr8k_text/Flickr8k.token.txt"


# Đọc file các caption
def load_doc(filename):
	# open the file as read only
	file = open(filename, 'r')
	# read all text
	text = file.read()
	# close the file
	file.close()
	return text

doc = load_doc(path_to_desc)

# Lưu caption dưới dạng key value: id_image : ['caption 1', 'caption 2', 'caption 3',' caption 4', 'caption 5']
def load_descriptions(doc):
	mapping = dict()
	# process lines
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		if len(line) < 2:
			continue
		# take the first token as the image id, the rest as the description
		image_id, image_desc = tokens[0], tokens[1:]
		# extract filename from image id
		image_id = image_id.split('.')[0]
		# convert description tokens back to string
		image_desc = ' '.join(image_desc)
		# create the list if needed
		if image_id not in mapping:
			mapping[image_id] = list()
		# store description
		mapping[image_id].append(image_desc)
	return mapping

descriptions = load_descriptions(doc)

# Hàm tiền xử lý các câu mô tả như: đưa về chữ thường, bỏ dấu, bỏ các chữ số....
def clean_descriptions(descriptions):
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for key, desc_list in descriptions.items():
		for i in range(len(desc_list)):
			desc = desc_list[i]
			# tokenize
			desc = desc.split()
			# convert to lower case
			desc = [word.lower() for word in desc]
			# remove punctuation from each token
			desc = [w.translate(table) for w in desc]
			# remove hanging 's' and 'a'
			desc = [word for word in desc if len(word)>1]
			# remove tokens with numbers in them
			desc = [word for word in desc if word.isalpha()]
			# store as string
			desc_list[i] =  ' '.join(desc)

clean_descriptions(descriptions)

# Lưu description xuống file
def save_descriptions(descriptions, filename):
	lines = list()
	for key, desc_list in descriptions.items():
		for desc in desc_list:
			lines.append(key + ' ' + desc)
	data = '\n'.join(lines)
	file = open(filename, 'w')
	file.write(data)
	file.close()

save_descriptions(descriptions, 'descriptions.txt')

################################ II. PHẦN ĐỌC ẢNH  ẢNH ############################
path_to_train_id_file = 'Flickr8k/Flickr8k_text/Flickr_8k.trainImages.txt'
path_to_test_id_file ='Flickr8k/Flickr8k_text/Flickr_8k.testImages.txt'
path_to_image_folder = 'Flickr8k/Flicker8k_Dataset/'

# Lấy id ảnh tương ứng với dữ liệu train, test, dev
def load_set(filename):
	doc = load_doc(filename)
	dataset = list()
	# process line by line
	for line in doc.split('\n'):
		# skip empty lines
		if len(line) < 1:
			continue
		# get the image identifier
		identifier = line.split('.')[0]
		dataset.append(identifier)
	return set(dataset)

train_id = load_set(path_to_train_id_file)

# 1. -------- Xử lý ảnh dùng để train -----------------------------------------------


# Lấy lấy các ảnh jpg trong thư mục chứa toàn bộ ảnh Flickr
all_train_images_in_folder = glob.glob(path_to_image_folder + '*.jpg')

# Đọc toàn bộ nội dung file danh sách các file ảnh dùng để train
train_images = set(open(path_to_train_id_file, 'r').read().strip().split('\n'))

# Danh sách ấy sẽ lưu full path vào biến train_img
train_img = []

for image in all_train_images_in_folder: # Duyệt qua tất cả các file trong folder
    if image[len(images):] in train_images: # Nếu tên file của nó thuộc training set
        train_img.append(image) # Thì thêm vào danh sách ảnh sẽ dùng để train

# 2. -------- Xử lý ảnh dùng để test (xử lý tương tự) -------------------------------------

test_images = set(open(path_to_train_id_file, 'r').read().strip().split('\n'))
test_img = []

for image in all_train_images_in_folder:
    if image[len(images):] in test_images:
        test_img.append(i)

############################ III. PHẦN XỬ LÝ DỮ LIỆU MÔ TẢ ############################

# Hàm đọc mô tả từ file  và Thêm 'startseq', 'endseq' cho chuỗi
def load_clean_descriptions(filename, dataset):
	# load document
	doc = load_doc(filename)
	descriptions = dict()
	for line in doc.split('\n'):
		# split line by white space
		tokens = line.split()
		# split id from description
		image_id, image_desc = tokens[0], tokens[1:]
		# skip images not in the set
		if image_id in dataset:
			# create list
			if image_id not in descriptions:
				descriptions[image_id] = list()
			# wrap description in tokens
			desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
			# store
			descriptions[image_id].append(desc)
	return descriptions

train_descriptions = load_clean_descriptions('descriptions.txt', train_id)

# Tạo list các training caption
all_train_captions = []
for key, val in train_descriptions.items():
    for cap in val:
        all_train_captions.append(cap)
len(all_train_captions)

# Chỉ lấy các từ xuất hiện trên 10 lần
word_count_threshold = 10
word_counts = {}
nsents = 0
for sent in all_train_captions:
    nsents += 1
    for w in sent.split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1

vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]

# Tạo từ điển map từ Word sang Index và ngược lại
ixtoword = {}
wordtoix = {}

ix = 1
for w in vocab:
    wordtoix[w] = ix
    ixtoword[ix] = w
    ix += 1

# Tính toán bảng từ vựng
vocab_size = len(ixtoword) + 1 # Thêm 1 cho từ dùng để padding

# Chuyển thành từng dòng
def to_lines(descriptions):
	all_desc = list()
	for key in descriptions.keys():
		[all_desc.append(d) for d in descriptions[key]]
	return all_desc

# Tính toán độ dài nhất của câu mô tả
def max_length(descriptions):
	lines = to_lines(descriptions)
	return max(len(d.split()) for d in lines)

max_length = max_length(train_descriptions)

############################ IV. PHẦN XỬ LÝ ẢNH ĐẦU VÀO ĐỂ ĐƯA VAO MODEL CHÍNH  ############################

# Hàm load ảnh, resize về khích thước mà Inception v3 yêu cầu.
def preprocess(image_path):
    # Resize ảnh về 299x299 làm đầu vào model
    img = image.load_img(image_path, target_size=(299, 299))
    x = image.img_to_array(img)
    # Thêm một chiều  nữa
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

# Khởi tạo Model INception v3 để tạo ra feauture cho các ảnh của chúng ta
model = InceptionV3(weights='imagenet')
model_new = Model(model.input, model.layers[-2].output)

# Hàm biến ảnh đầu vào thành vector features (2048, )
def encode(image):
    image = preprocess(image) # preprocess the image
    fea_vec = model_new.predict(image) # Get the encoding vector for the image
    fea_vec = np.reshape(fea_vec, fea_vec.shape[1]) # reshape from (1, 2048) to (2048, )
    return fea_vec

# Lặp một vòng qua các ảnh train và biến hết thành vector features
encoding_train = {}
for img in train_img:
    encoding_train[img[len(images):]] = encode(img)

# Lưu image embedding train lại
with open("encoded_train_images.pkl", "wb") as encoded_pickle:
    dump(encoding_train, encoded_pickle)

# Lặp một vòng qua các ảnh test và biến hết thành vector features
encoding_test = {}
for img in test_img:
    encoding_test[img[len(images):]] = encode(img)
with open("encoded_test_images.pkl", "wb") as encoded_pickle:
    dump(encoding_test, encoded_pickle)

train_features = load(open("encoded_train_images.pkl", "rb"))

############################ V. PHẦN XỬ LÝ MÔ TẢ ĐẦU VÀO ĐỂ ĐƯA VAO MODEL CHÍNH  ############################

# Tải model Glove để embeding word
glove_dir = ''
embeddings_index = {} # empty dictionary
f = open(os.path.join(glove_dir, 'glove.6B.200d.txt'), encoding="utf-8")

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

embedding_dim = 200
# Tạo ma trận embeding cho bảng từ vững, mỗi từ embeding bằng 1 vector 200
embedding_matrix = np.zeros((vocab_size, embedding_dim))

# Lặp qua các từ trong danh sách từ
for word, i in wordtoix.items():
    # Lấy embeding của Glove gán vào embeding vector
    embedding_vector = embeddings_index.get(word)
    # Nếu như không None thì gán vào mảng Maxtrix
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

############################ VI. TẠO MODEL CHÍNH VÀ TIẾN HÀNH TRAIN  ############################

# Tạo model chính

# Nhánh 1. Input image vector
inputs_image = Input(shape=(2048,))
dr_1 = Dropout(0.5)(inputs_image)
fc_1 = Dense(256, activation='relu')(drop_1)

# Nhánh 2. Input câu mô tả
inputs_desc = Input(shape=(max_length,))
emb_1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs_desc)
dr_2 = Dropout(0.5)(emb_1)
lstm_1 = LSTM(256)(dr_2)

# Gộp 2 input
decoder_1 = add([fc_1, lstm_1])
fc_3 = Dense(256, activation='relu')(decoder_1)

# Layer output
outputs = Dense(vocab_size, activation='softmax')(fc_3)
model = Model(inputs=[inputs_image, inputs_desc], outputs=outputs)

# Layer 2 dùng GLOVE Model nên set weight thẳng và không cần train
model.layers[2].set_weights([embedding_matrix])
model.layers[2].trainable = False

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam')

#  Hàm sinh dữ liệu train để feed cho Model
def data_generator(descriptions, photos, wordtoix, max_length, num_photos_per_batch):
    X1, X2, y = list(), list(), list()
    n=0
    # loop for ever over images
    while 1:
        for key, desc_list in descriptions.items():
            n+=1
            # retrieve the photo feature
            photo = photos[key+'.jpg']
            for desc in desc_list:
                # encode the sequence
                seq = [wordtoix[word] for word in desc.split(' ') if word in wordtoix]
                # split one sequence into multiple X, y pairs
                for i in range(1, len(seq)):
                    # split into input and output pair
                    in_seq, out_seq = seq[:i], seq[i]
                    # pad input sequence
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    # encode output sequence
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    # store
                    X1.append(photo)
                    X2.append(in_seq)
                    y.append(out_seq)
            # yield the batch data
            if n==num_photos_per_batch:
                yield [[array(X1), array(X2)], array(y)]
                X1, X2, y = list(), list(), list()
                n=0

# Cài đặt tham số train và TRAIN!!!
model.optimizer.lr = 0.0001
epochs = 10 # Số epoch
number_pics_per_batch = 6 # Số lượng ảnh mỗi batch
steps = len(train_descriptions)//number_pics_per_batch # Số bước mỗi epoch

# Train nào
for i in range(epochs):
    generator = data_generator(train_descriptions, train_features, wordtoix, max_length, number_pics_per_bath)
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)

# Sau 10 bước train ta lưu weights
model.save_weights('model_30.h5')

############################ VII. KIỂM THỬ XEM MODEL HOẠT ĐỘNG NHƯ NÀO  ############################

# Nạp list các feature của ảnh test (đã encode bằng Inception ở bước trên
with open("encoded_test_images.pkl", "rb") as encoded_pickle:
    encoding_test = load(encoded_pickle)

# Hàm đặt Caption
# Với môi ảnh mới khi test, ta sẽ bắt đầu chuỗi với 'startseq' rồi sau đó cho vào model để dự đoán từ tiếp theo. Ta thêm từ
# vừa được dự đoán vào chuỗi và tiếp tục cho đến khi gặp 'endseq' là kết thúc hoặc cho đến khi chuỗi dài 34 từ.
def setCaption(photo):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = [wordtoix[w] for w in in_text.split() if w in wordtoix]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = ixtoword[yhat]
        in_text += ' ' + word
        if word == 'endseq':
            break
    final = in_text.split()
    final = final[1:-1]
    final = ' '.join(final)
    return final

# Dự đoán thử 1 ảnh
image_test_id =5 # Chọn thử anh số 5

# Lấy tên của ảnh só 5 đó
image_name = list(encoding_test.keys())[image_test_id]

# Convert thành vector 2048
image_vector = encoding_test[pic].reshape((1,2048))

# Vẽ ảnh lên màn hình
x=plt.imread(path_to_image_folder + image_name)
plt.imshow(x)
plt.show()

# In kết quả mô tả do model tự đặt cho ảnh
print(setCaption(image))