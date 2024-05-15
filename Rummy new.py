#!/usr/bin/env python
# coding: utf-8

# # Data Generation for the Rummy Pool

# In[1]:


import random
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl



numbers_of_players_in_pool = int(input('How many players do you want to have in the pool: '))
bet_multiplier_options = [0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 40, 80, 125, 200]
min_amount = [4, 8, 20, 40, 80, 160, 240, 400, 800, 1600, 3200, 6400, 10000, 16000]
bet_min = dict(zip(bet_multiplier_options, min_amount))
print(bet_min)

bet_multiplier_input = float(input("How much do you want the bet amount multiplier to be?\nChoose from the options - (0.05, 0.1, 0.25, 0.5, 1, 2, 3, 5, 10, 20, 40, 80, 125, 200): "))
print(f"You have chosen the bet amount multiplier to be: {bet_multiplier_input}")

for key in bet_min.keys():
    if key == bet_multiplier_input:
        min_amt_required = float(bet_min[key])
        print("Minimum amount is:", min_amt_required)
        print(type(min_amt_required))
    else:
        continue
        print("Sorry, you have chosen a wrong bet amount multiplier, please try again by chosing one from the options mentioned here (0.05, 0.1, 0.25, 0.5, 1, 2, 3,5, 10, 20, 40, 80, 125, 200): ")

totalPlayers = {}
for i in range(numbers_of_players_in_pool):
    x = ''.join([random.choice(string.ascii_uppercase) for k in range(4)]).title()
    y = round(random.uniform(min_amt_required, (min_amt_required + 30)), 3)
    totalPlayers[x] = y

Sum = sum(totalPlayers.values())
print(f"The total amount involved in this pool is: {Sum}")
total_players_df = pd.DataFrame(totalPlayers.items(), columns=['Name', 'Balance in Rupees'])
print("The below table shows all the players in the pool along with their account balances.")
print(total_players_df)

total_amount_credited_to_app = 0
percentage_deduction = 15 / 100



matches_data = []

def matches():
    global total_amount_credited_to_app
    global matches_data

    eligible_players = {k: v for k, v in totalPlayers.items() if v >= min_amt_required}

    if len(eligible_players) < 2:
        print("Insufficient eligible players for a match.")
        return

    p1 = random.choice(list(eligible_players.items()))
    del eligible_players[p1[0]]
    p2 = random.choice(list(eligible_players.items()))

    match_data = {
        'Match Number':'',
        'Player 1': p1[0],
        'Player 2': p2[0],
        'Loser': '',
        'Winner': '',
        'Points Lost by Loser': '',
        'Amount Lost by Loser': '',
        'Amount Remaining for Loser': '',
        'Amount to be Credited to App': '',
        'Total Amount Earned by App': '',
        'Amount Won by Winner': '',
        'Amount Remaining for Winner': ''
    }
    
    match_data['Match Number'] = len(matches_data) + 1
    
    remaining_balances = totalPlayers.copy()
    for player, balance in remaining_balances.items():
        if player == match_data['Loser']:
            match_data[player] = totalPlayers[player] - (random.randint(2, 80) * bet_multiplier_input)
        elif player == match_data['Winner']:
            match_data[player] = totalPlayers[player] + (
                        totalPlayers[match_data['Loser']] * bet_multiplier_input * (100 - percentage_deduction) / 100)
        else:
            match_data[player] = balance
    
    for players, remaining_bal in totalPlayers.items():
        if players == match_data['Loser']:
            match_data[players] = amount_remaining_for_loser
        elif players == match_data['Winner']:
            match_data[players] == amount_remaining_for_winner
        else:
            match_data[players] == remaining_bal
    
    match_data['Loser'] = p1[0] if random.choice([p1, p2]) == p2 else p2[0]
    match_data['Winner'] = p1[0] if match_data['Loser'] == p2[0] else p2[0]

    points_lost = random.choice(range(2, 80))
    amount_deducted_from_loser = round(points_lost * bet_multiplier_input, 3)
    amount_remaining_for_loser = round(totalPlayers[match_data['Loser']] - amount_deducted_from_loser, 3)
    amount_credited_to_app = round(amount_deducted_from_loser * percentage_deduction, 3)
    total_amount_credited_to_app += amount_credited_to_app

    match_data['Points Lost by Loser'] = points_lost
    match_data['Amount Lost by Loser'] = amount_deducted_from_loser
    match_data['Amount Remaining for Loser'] = amount_remaining_for_loser
    match_data['Amount to be Credited to App'] = amount_credited_to_app
    match_data['Total Amount Earned by App'] = total_amount_credited_to_app

    amount_earnt_by_winner = round(amount_deducted_from_loser - amount_credited_to_app, 3)
    amount_remaining_for_winner = round(totalPlayers[match_data['Winner']] + amount_earnt_by_winner, 3)

    match_data['Amount Won by Winner'] = amount_earnt_by_winner
    match_data['Amount Remaining for Winner'] = amount_remaining_for_winner

    totalPlayers[match_data['Loser']] = amount_remaining_for_loser
    totalPlayers[match_data['Winner']] = amount_remaining_for_winner
    
    
    print(f"In this match we have : \n {p1} v/s {p2}")
    print(f'Where player 1 is : {p1}')
    print(f'and player 2 is : {p2}')
    
    print (f"The Winner is : {match_data['Winner']} ")
    print(f"And the loser is : {match_data['Loser']}  ")
    
    print(f"Points lost by loser - {match_data['Loser']}  is : \n {points_lost}")
    print(f"Amount lost by loser - {match_data['Loser']}  is : \n {amount_deducted_from_loser}")
    print (f"Amount remaining in the loser {match_data['Loser']}'s account is : \n {amount_remaining_for_loser}")
    (f"Amount won by winner {match_data['Winner']} is : \n {amount_earnt_by_winner}")
    print(f"Amount remaining in the winner {match_data['Winner']}'s account is : \n {amount_remaining_for_winner}")
    
    print(f"Amount to be credited to app is : \n {amount_credited_to_app}")
    print(f"Total amount earnt by app is : \n {total_amount_credited_to_app}")
    
    closing_balance_of_match = pd.DataFrame(totalPlayers.items(), columns=['Name', "Balance"])
    closing_balance_of_match.loc[closing_balance_of_match['Name'] == match_data['Loser'], 'Balance'] = amount_remaining_for_loser
    closing_balance_of_match.loc[closing_balance_of_match['Name'] == match_data['Winner'], 'Balance'] = amount_remaining_for_winner
    print(f"The closing balance of this match is \n {closing_balance_of_match}")

   
    matches_data.append(match_data)

counter = len(totalPlayers)
print(counter)
match = 0

while counter > 1:
    matches()
    match += 1
    print(f"Match {match} concluded")
    print("-" * 120)
    counter = sum(1 for amount in totalPlayers.values() if amount >= min_amt_required)

matches_df = pd.DataFrame(matches_data)



# In[2]:


matches_df


# In[3]:


matches_df.to_excel('Rummy.xlsx', index=False, engine='openpyxl')


# In[4]:


App = matches_df['Total Amount Earned by App']
col_names = matches_df.columns


# ### Visualization of player's and app's flow of balances

# In[5]:


font = {'size':24, 'family':'Palatino Linotype', 'weight':'bold'}
for i in range(numbers_of_players_in_pool):
    plt.figure(figsize=(25,5))
    plott = sns.histplot(matches_df.iloc[:,12+i], kde=True, bins=40, color= random.choice(['r', 'g','b','yellow','black','orange']))
    plt.xlabel(f"{col_names[12+i]}'s flow of balance", fontdict=font)
    plt.ylabel(f"Graph {i+1}", fontdict=font)
    plt.show()
    i += 1
#     if i > len(matches_df.columns):
#         exit()


# In[6]:


plt.figure(figsize=(25,5))
sns.histplot(matches_df.iloc[:,9], kde=True, bins=40, color='black')
plt.xlabel("App's flow of balance", fontdict=font)
plt.ylabel("Count", fontdict=font)


# In[7]:


for i in range(numbers_of_players_in_pool):
    plt.figure(figsize=(25,5))
    matches_df.iloc[:,12+i].plot(kind='line', color= random.choice(['r','g','b','c','black','indigo','orange']))
    plt.xlabel(f"{col_names[12+i]}'s flow of balance", fontdict=font)
    plt.ylabel(f"Graph {i+1}", fontdict=font)
    plt.show()
    i += 1


# In[8]:


plt.figure(figsize=(25,5))
matches_df['Total Amount Earned by App'].plot(kind='line', color='indigo')
plt.xlabel("App's flow of balance", fontdict=font)
plt.ylabel("Count", fontdict=font)


# ### Linear Regression model preparation

# In[9]:


predictor = matches_df.drop('Total Amount Earned by App', axis=1)


# In[10]:


predictor


# In[11]:


target = pd.DataFrame(matches_df['Total Amount Earned by App'])
target


# In[12]:


cat_cols = [col for col in predictor.columns.values if predictor[col].dtype == 'object']
predictor_cat = predictor[cat_cols]
predictor_cat


# In[13]:


cat_dummy = pd.get_dummies(predictor_cat, drop_first=True)
cat_dummy


# In[14]:


predictor_num = predictor.drop(cat_cols, axis=1)
predictor_num


# In[15]:


predictor_num['Match Number'] = predictor_num['Match Number'].astype('float64')
predictor_num['Points Lost by Loser'] = predictor_num['Points Lost by Loser'].astype('float64')


# ### Logarithmic Transformation to convert the target's unifrom distribution to a normal distribution

# In[16]:


import numpy as np

target_log = np.log1p(target)


# In[17]:


target_log


# In[18]:


sns.histplot(target, kde=True)


# In[19]:


sns.histplot(target_log, kde=True)


# In[20]:


cat_dummy.columns


# In[21]:


predictor_new = pd.concat([cat_dummy, predictor_num], axis =1)
predictor_new


# In[22]:


from sklearn.model_selection import train_test_split

X_train, x_test, Y_train, y_test = train_test_split(predictor_new.astype(float), target_log.astype(float), test_size=0.3, random_state=2024)


# In[23]:


X_train.shape


# In[24]:


x_test.shape


# In[25]:


Y_train.shape


# In[26]:


y_test.shape


# In[27]:


from sklearn.linear_model import LinearRegression

lin = LinearRegression()

model = lin.fit(X_train, Y_train)


# In[28]:


import statsmodels.api as sm

#X_train = sm.add_constant(X_train)

model = sm.OLS(Y_train, X_train).fit()


# In[29]:


model.summary()


# In[30]:


y_pred = model.predict(x_test)


# In[31]:


y_pred


# In[32]:


pd.set_option('display.max_rows',150)


# In[33]:


matches_df.isnull().sum().head(112)


# In[34]:


new = pd.concat([predictor_num,target], axis=1)


# In[35]:


new


# In[36]:


new.columns


# In[37]:


new_1 = new.iloc[:,7:]


# In[38]:


new_1


# In[39]:


for col in new_1.columns.values:
    new_1[f'{col} loss'] = new_1[col].diff()


# In[40]:


new_1


# #### Handling missing values

# In[41]:


new_1.isnull().sum()


# In[42]:


new_1[new_1.columns[4:]] = new_1[new_1.columns[4:]].fillna(0)


# In[43]:


new_1


# In[44]:


# plt.figure(figsize=(10,5))
# sns.heatmap(new_1.corr(), annot=True, cmap='coolwarm')


# In[45]:


new = new.drop(['Match Number', 'Points Lost by Loser', 'Amount Lost by Loser','Amount Remaining for Loser', 'Amount Won by Winner', 'Amount Remaining for Winner'], axis=1)


# In[46]:


# plt.figure(figsize=(15,10))
# sns.heatmap(new.corr(numeric_only=True), annot=True, fmt='0.2g', cmap='viridis', cbar=True, square=True)


# In[47]:


font1 = {'size':12, 'family':'Palatino Linotype', 'weight':'bold'}
matches_df.plot(x=random.choice(matches_df.columns.values[12:]), legend=False, y='Total Amount Earned by App', kind='line',  color=random.choice(['r','g','b','yellow','black','orange','indigo']))
plt.title("Player's flow of balance as compared to the App's flow of balance", fontdict=font1)
plt.ylabel("App")

#Change value from 12 to any greater int value within the length of columns.values to get similar visualizations for other players


# In[58]:


df = pd.read_csv(r"D:\DSP and ML files\CSV\Rummy 4 Player 0.5.csv")
#Save the csv file in local storage and access it in the above code


# In[59]:


df.head()


# In[60]:


df.describe().transpose()


# In[61]:


df['Losing Players Balance 1'].plot(kind='line')


# In[62]:


df['Total Money Involved in each pool'] = df[['Losing Players Balance 1', 'Losing Players Balance 2', 'Losing Players Balance 3', 'Winning Player Balance', 'Total Amount Earned by App']].sum(axis=1)


# In[63]:


df.columns


# In[64]:


df.head()


# In[67]:


plt.figure(figsize=(5,5))

sns.displot(data=df, x='Total Money Involved in each pool'), sns.displot(data=df, x='Total Amount Earned by App', color='black')


# In[ ]:




