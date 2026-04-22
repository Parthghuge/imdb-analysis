import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

print("IMDB Analysis")

# load data
df = pd.read_csv("imdb.csv")

print(df.head())

# -------------------------
# basic info
# -------------------------

print("\nTotal movies:", len(df))
print("Average rating:", round(df["IMDB_Rating"].mean(), 2))


# -------------------------
# top rated movies (FIXED)
# -------------------------

top = df.sort_values("IMDB_Rating", ascending=False).head(10)

print("\nTop Movies:")
print(top[["Series_Title", "IMDB_Rating"]])

plt.figure(figsize=(10,5))
plt.barh(top["Series_Title"], top["IMDB_Rating"])
plt.title("Top Rated Movies")
plt.xlabel("Rating")
plt.ylabel("Movie")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()


# -------------------------
# genre analysis
# -------------------------

g = df["Genre"].value_counts().head(10)

plt.figure()
g.plot(kind="bar")
plt.title("Top Genres")
plt.xticks(rotation=45)
plt.show()


# -------------------------
# rating distribution
# -------------------------

plt.figure()
df["IMDB_Rating"].plot(kind="hist", bins=10)
plt.title("Ratings Distribution")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()


# -------------------------
# votes vs rating (regression)
# -------------------------

data = df.dropna(subset=["IMDB_Rating", "No_of_Votes"])

X = data[["No_of_Votes"]]
y = data["IMDB_Rating"]

model = LinearRegression()
model.fit(X, y)

pred = model.predict(X)

plt.figure()
plt.scatter(X, y)
plt.plot(X, pred)
plt.title("Votes vs Rating")
plt.xlabel("Votes")
plt.ylabel("Rating")
plt.show()


# -------------------------
# directors
# -------------------------

d = df["Director"].value_counts().head(10)

plt.figure()
d.plot(kind="bar")
plt.title("Top Directors")
plt.xticks(rotation=45)
plt.show()


# -------------------------
# year trend
# -------------------------

year = df.groupby("Released_Year")["IMDB_Rating"].mean()

plt.figure()
year.plot()
plt.title("Average Rating by Year")
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()


print("\nDone")
