#getting started with pandas
import pandas as pd
mydataset = {
  'cars': ["BMW", "Volvo", "Ford"],
  'passings': [3, 7, 2]
}
myvar = pd.DataFrame(mydataset)
print("This is the dataframe\n", myvar)



#creating a series: a series is like a column in a table
a = [1, 7, 2]
myser = pd.Series(a)
print("This is the series\n",myser)

#visualizing labels: labels are like the name of a row in a table
print("This is the series with labels\n",pd.Series(a, index = ["x", "y", "z"])) 


#reading a csv file
df = pd.read_csv('data.csv')
print("This is the csv file\n",df)
print(pd.options.display.max_rows) 

#reading json file
data = pd.read_json('data.json')
print("This is the json file\n",data.to_string())



#analzying data
# head will all the series and the first n rows
print("The head function \n", df.head(10))  #first 5 rows
print("This is the data description\n", df.describe())
print("This is the tail", df.tail(3))  #last 3 rows

print("df.info()\n", df.info())

