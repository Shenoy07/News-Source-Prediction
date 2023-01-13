library(httr)
library(tidyverse)

offset = 0
sources<- c("cnn","bbc","guardian","nytimes","aljazeera","espn","sportskeeda","dailymail")
categories<- c("general","business","entertainment","health","science","sports","technology")
columns <- c("title","description","category","source")
myData = data.frame(matrix(nrow=0,ncol=length(columns)))
colnames(myData) = columns
base_url ="http://api.mediastack.com/v1/news?access_key=57c918db6827de92d84955d2e9ae2669&languages=en&limit=100&date=2014-12-24,2022-12-31"
for (source in sources){
  for (category in categories){
    for (x in 1:300){
      modified_url = paste0(base_url,"&sources=",source,"&categories=",category,"&offset=",offset)
      # print(modified_url)
      response = GET(url = modified_url)
      response_content = content(response)
      data = response_content$data
      print(length(data))
      print(source)
      print(category)
      if (length(data) == 0){
        offset = 0
        break
      }
      df = map_df(data,magrittr::extract,c("title","description","category","source"))
      myData = rbind(myData,df)
      offset = offset + 100
    }
  }
}
myData <- unique(myData)

write.csv(myData,"myDataBaseFinal1.csv", row.names = FALSE)

