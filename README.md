# Data_Mining_Project_ACDC

## Interactive member dashboard

The Streamlit dashboard provides filters for country, industry, plan tier, gender,
account type, membership status, and age. Its overview shows attribute distributions
and member mix; the segment explorer drills into a selected group and lets you
download the matching records as CSV.

Install the project dependencies and start the dashboard from the repository root:

```powershell
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

The dashboard reads the dataset configured in `config.yaml`, removes exact duplicate
rows, standardizes configured categories and boolean fields, and preserves numeric
values even when they are outside the configured validity ranges. The **Data quality**
tab reports out-of-range values, date anomalies (including retained sentinel dates),
and non-empty values that could not be parsed, without changing the displayed data.

Consumers today are becoming more selective about the business they support and where they spend
their money. This makes it vital for companies to develop a deep understanding of their customer base in
order to tailor their services and marketing strategies more effectively.

Amazing Creative Desk Coworking (ACDC) faces exactly this challenge with its diverse member base. In
a competitive coworking industry, understanding member segments is key to improving satisfaction,
retention, and revenue. This process, known as customer segmentation, allows ACDC to divide
customers into smaller groups that share similar characteristics or behaviors. For instance, some members
prioritize premium services and dedicated space, while others are cost-conscious and focused on basic
desk access. By uncovering these patterns, ACDC can tailor its services and membership perks to meet
the unique needs and behaviors of each segment.

To do:
K: gender/industry/country
Mi: Date columns/redundancies
Ma: feature engineering/slides
F: feature engineering/lifetime value

group task: answer the 4 questions
1. Which findings were most unexpected or insightful, and what do they reveal about likely customer clusters? 
2. What data limitations pose the greatest risks for clustering, and how might they be mitigated? 
3. Which patterns in customer activity, including those revealed by engineered features, suggest natural 
groupings, and what cluster characteristics do you anticipate? 
4. How would you communicate these insights to non-technical stakeholders? Include a clear explanation of 
the expected number of clusters, the most important differentiating features, and any anticipated 
challenges. 
