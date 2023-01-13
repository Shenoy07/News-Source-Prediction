# News-Source-Prediction

## Business Context

We are using mediastack REST API interface which provides worldwide live and historical news data. The API provides multiple parameters such as 'Category', 'language','source',etc to get the required news articles.

## Problem Description

When we go through a news article online, we are always concerned whether our news is coming from the right source.

Media stack has news articles from multiples sources/channels. Our goal is to train ML models that can predict the source of the news based on the content provided.

**Project Files:**

All the relevant project files are found in this link: <https://drive.google.com/drive/folders/1C0D_GeLsS2E1LChlnG7gbXlXFebWsSQz?usp=share_link>

```{r}
library(httr)
library(tidyverse)
library(repurrrsive)
library(stringr)
library(dplyr)
library(RColorBrewer)
library(ggthemes)
library(keras)
library(tokenizers)
library(kableExtra)
```

**DATA FILE:**

1.  The code for extraction of data from the API is in the datacolllection.R file, please download it, change the API access key and run it to get the file below.

**OR**

2.  Just download the myDataBaseFinal1.csv and paste it in directory and run the program.

## Download csv file of data collected from the API

```{r}
data <- read.csv("myDataBaseFinal1.csv")
```

Now we do some cleaning as required, so that it doesn't cause issues while modelling.

Here were are adjusting our source names so keep the regularity.

```{r}
data <- data.frame(lapply(data, function(x) {
  gsub("The New York Times", "nytimes", x)}))
data <- data.frame(lapply(data, function(x) {
  gsub("Al Jazeera", "aljazeera", x)}))
data <- data.frame(lapply(data, function(x) {
  gsub("The Guardian", "guardian", x)}))
```

```{r}
data[c('Source', 'Source2')] <- str_split_fixed(data$source, ' ', 2)
df = subset(data, select = -c(source,Source2) )
names(df)[4] <- "source"
```

## Filtering News that are longer than 100 words

```{r}
desired_length = 100
df = data %>% 
  filter(str_count(description) >= desired_length) %>%
  select(title:source)
```

Now the final count in our dataset is as follows:

```{r}
dim(df)
```

## Data Summary

```{r}
dt <- head(df)
dt %>%
  kbl() %>%
  kable_styling()
```
![image](https://user-images.githubusercontent.com/31709147/212264549-488a49a2-b8c6-42b4-aff5-86fb805af3d0.png)

The Data has 4 columns which are:

-   Title
-   Description
-   Category
-   Source

For our modeling we will be using Description to predict the source of news

## Data Exploration

Below are the different news sources and the counts of their respective articles in our dataset

```{r}
df3 <- df %>%
  count(source)
names(df3)[2] <- "count"
ggplot(data=df3, aes(fill=source,y=count, x=source)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_text(aes(label=count),vjust = -0.2, position = position_dodge(.9))+
  theme_gdocs()+
  scale_color_gdocs()+
  scale_fill_brewer(palette = "Paired")+
  theme(axis.text.x = element_text(angle = 90, 
                                               vjust = 0.5, 
                                               hjust=1))
```
![image](https://user-images.githubusercontent.com/31709147/212264660-53913caf-191c-4b69-a10b-e08d86b1f958.png)

The plot below shows the counts of different news categories for each news source

```{r}
df2 <- df %>%
  group_by(source,category)%>%
  count(source)
names(df2)[3] <- "count"
ggplot(data=df2, aes(fill=category, y=count, x=source)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_text(aes(label=count),vjust = -0.2, position = position_dodge(.9))+
  theme_gdocs()+
  scale_color_gdocs()+
  scale_fill_brewer(palette = "Paired")+
  theme(axis.text.x = element_text(angle = 90, 
                                               vjust = 0.5, 
                                               hjust=1))
```
![image](https://user-images.githubusercontent.com/31709147/212264727-ac742c6b-cfb9-4232-904f-b1713e32489b.png)

The final data exploratory plot shows the length of news article descriptions in our dataset

```{r}
df_length <- df%>%
  mutate(description_length = str_count(description))
vec1 <- df_length$description_length 
DT <- df_length %>% 
  group_by(gr=cut(vec1, breaks= seq(0, 650, by = 50)) ) %>% 
  summarise(n= n()) %>%
  arrange(as.numeric(gr))
names(DT)[1] <- "Length_Bin"
names(DT)[2] <- "Count"
ggplot(data=DT, aes(fill = Length_Bin, y=Count, x=Length_Bin)) + 
  geom_bar(position="dodge", stat="identity")+
  geom_text(aes(label=Count),vjust = -0.2, position = position_dodge(.9))+
  theme_gdocs()+
  scale_color_gdocs()+
  theme(axis.text.x = element_text(angle = 90, 
                                               vjust = 0.5, 
                                               hjust=1))
![image](https://user-images.githubusercontent.com/31709147/212264811-f8599993-230e-418f-aab2-853257c1cdc1.png)

myData <- df
```

## AI/ML Procedure Summary

defining the default parameters

```{r}
max_words <- 10000
maxlen <- 400
training_samples <- as.integer(0.75 * dim(myData)[1])
validation_samples <- as.integer(0.25 * dim(myData)[1])
tokenizer <- text_tokenizer(num_words = max_words) %>% fit_text_tokenizer(myData$description)
sequences <- texts_to_sequences(tokenizer, myData$description)
word_index = tokenizer$word_index
cat("Found",length(word_index), "unique tokens. \n")
```

padding sequences to make them of equal length

```{r}
data <- pad_sequences(sequences, maxlen = maxlen)
labels <- as.array(as.numeric(factor(myData$source)))
cat("Shape of data tensor (Num Docs, Num Words in a Doc):", dim(data), "\n")
cat('Shape of label tensor (Num Docs):', dim(labels), "\n")
```

```{r}
unique(labels)
length(unique(labels))
```

Training and Validation split

```{r}
set.seed(123)
indices <- sample(1:nrow(data))
training_indices <- indices[1:training_samples]
validation_indices <- indices[(training_samples + 1):
                                (training_samples + validation_samples)]
x_train <- data[training_indices,]
y_train <- labels[training_indices]
x_val <- data[validation_indices,]
y_val <- labels[validation_indices]
y_val<- to_categorical(y_val)
y_train <- to_categorical(y_train)
dim(x_train)
dim(y_train)
dim(x_val)
dim(y_val)
```

## nn_embd

```{r}
embedding_sizes = c(100,200,300,400)
result_list = c()
for (embedding in embedding_sizes){
  print(embedding)
  embedding_dim <- embedding
  model <- keras_model_sequential() %>% 
    layer_embedding(input_dim = max_words,
                    input_length = maxlen,
                    output_dim = embedding_dim) %>%
    layer_flatten() %>%
    layer_dense(units = 32, activation = "relu") %>%
    layer_dense(units = dim(y_train)[2], activation = "softmax")
  
  model %>% compile(
    optimizer = "adam",
    loss = "categorical_crossentropy",
    metrics = c("accuracy")
  )
  
  history <- model %>% fit(
    x_train, y_train,
    epochs = 20,
    batch_size = 256,
    validation_data = list(x_val, y_val)
  )
  model %>% save_model_hdf5("nn_100.h5")
  print(plot(history))
  results <- model %>% evaluate(x_val, y_val)
  print(results)
  result_list = append(result_list,results)
}
```

## rnn model

```{r}
output_dim = 32
embedding_dim = 100
model_rnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim) %>%
  layer_simple_rnn(units = output_dim) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_rnn %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_rnn %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_rnn <- model_rnn %>% evaluate(x_val, y_val)
print(results_rnn)
result_list = append(result_list,results_rnn)
model_rnn %>% save_model_hdf5("rnn.h5")
```

## Multilayer RNN

```{r}
output_dim = 32
embedding_dim = 100
model_mlrnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim) %>%
  layer_simple_rnn(units = output_dim, return_sequences = TRUE) %>%
  layer_simple_rnn(units = output_dim, return_sequences = TRUE) %>%
  layer_simple_rnn(units = output_dim, return_sequences = TRUE) %>%
  layer_simple_rnn(units = output_dim) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_mlrnn %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_mlrnn %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_mlrnn <- model_mlrnn %>% evaluate(x_val, y_val)
print(results_mlrnn)
result_list = append(result_list,results_mlrnn)
model_mlrnn %>% save_model_hdf5("mlrnn.h5")
```

## LSTM

```{r}
output_dim = 32
embedding_dim = 100
model_lstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim) %>%
  layer_lstm(units = output_dim) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_lstm %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_lstm %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_lstm <- model_lstm %>% evaluate(x_val, y_val)
results_lstm
result_list = append(result_list,results_lstm)
model_lstm %>% save_model_hdf5("lstm.h5")
```

## Bi-directional LSTM

```{r}
output_dim = 32
embedding_dim = 100
model_bilstm <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim) %>%
  bidirectional(layer_lstm(units = output_dim)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_bilstm %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_bilstm %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_bilstm <- model_bilstm %>% evaluate(x_val, y_val)
results_bilstm
result_list = append(result_list,results_bilstm)
model_bilstm %>% save_model_hdf5("bilstm.h5")
```

## 1d convo

```{r}
filter_size = 32
embedding_dim = 128
model_1dcnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_max_pooling_1d(pool_size = 5) %>%
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>%
  layer_global_max_pooling_1d() %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_1dcnn %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_1dcnn %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_1dcnn <- model_1dcnn %>% evaluate(x_val, y_val)
results_1dcnn
result_list = append(result_list,results_1dcnn)
model_1dcnn %>% save_model_hdf5("1dcnn.h5")
```

## 1d convo with bi-lstm

```{r}
filter_size = 32
embedding_dim = 128
model_bilstm_1dcnn <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_words, output_dim = embedding_dim, input_length = maxlen) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu", input_shape = list(NULL, dim(data)[[-1]])) %>%
  layer_max_pooling_1d(pool_size = 3) %>%
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>%
  bidirectional(layer_lstm(units = 32)) %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = dim(y_train)[2], activation = "softmax")
model_bilstm_1dcnn %>% compile(
  optimizer = "adam",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
history <- model_bilstm_1dcnn %>% fit(
  x_train, y_train,
  epochs = 15,
  batch_size = 256,
  validation_split = 0.2
)
plot(history)
results_bilstm_1dcnn <- model_bilstm_1dcnn %>% evaluate(x_val, y_val)
results_bilstm_1dcnn
result_list = append(result_list,results_bilstm_1dcnn)
model_bilstm_1dcnn %>% save_model_hdf5("bilstm_1dcnn.h5")
```

## Creating Dataframe of all model accuracies

```{r}
models = c('NN_embd_100','NN_embd_200','NN_embd_300','NN_embd_400','RNN','Mulilayer RNN','LSTM','BI-LSTM','1d-CNN','1d-CNN-BiLSTM')
loss = c()
accuracy = c()
i=1
for (item in result_list){
  if (i%%2==1){
    loss = append(loss,item)
  }
  else{
    accuracy = append(accuracy,item)
  }
  i=i+1
}
# print(loss)
# print(accuracy)
result_df = data.frame(models = models, loss = loss, accuracy = accuracy)
write.csv(result_df,'results.csv')
```

## Model Evaluation Summary

```{r}
ggplot(data=result_df, aes(fill = models,y = accuracy, x = models)) + 
  geom_bar(position="dodge", stat="identity")+
  theme_gdocs()+
  scale_color_gdocs()+
  scale_fill_brewer(palette = "Paired")
```

1.  We have tried multiple combinations of deep learning neural networks such as LSTM, biLSTM & CNN.
2.  We have also tried glove embedding, the code for it is present in this google colab notebook, <https://colab.research.google.com/drive/1Sg5IdCBNjy1eYdwVDPLQ9xXfUnNdxkTC?usp=sharing>
3.  As you can see from the graph the best performance is achieved by Keras Model with embedding size 100, i.e. **NN_embd_100**
