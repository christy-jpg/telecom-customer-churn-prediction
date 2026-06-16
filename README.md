# Optimizing Telecom Customer Retention: A Predictive Machine Learning Approach
**Domain:** Telecommunications & Subscription Marketing  
**Methodology:** Binary Classification, Predictive Modeling, Feature Importance Analysis  

---

## 1. Executive Summary & Business Context

In the highly competitive telecommunications industry, customer acquisition costs (CAC) drastically exceed customer retention costs. Acquiring a new subscriber can cost up to 5 times more than keeping an existing one satisfied. For a telecom company, predicting customer churn before it happens is the single most effective way to protect recurring revenue streams and stabilize profit margins.

### Objective
The goal of this project is twofold:
1. Build a robust machine learning pipeline using Python and Scikit-Learn to accurately predict which customers are at risk of canceling their service (`Churn = 1`).
2. Extract mathematical feature importances to uncover the core behavioral triggers behind churn, transforming raw data into automated, proactive marketing strategies.

---

## 2. Dataset Overview & Feature Engineering

The model was trained on a dataset of **3,333 unique customer accounts** tracking 10 key behavioral, usage, and billing attributes. The dataset contains zero missing values and exhibits a baseline churn rate of approximately **14.5%** (483 churned accounts out of 3,333 total).

### Core Customer Attributes Analysed:
*   `Churn`: Target variable (1 = Cancelled service, 0 = Active/Retained)
*   `AccountWeeks`: Total tenure of the customer account in weeks
*   `ContractRenewal`: Binary indicator of recent contract renewal status
*   `DataPlan` & `DataUsage`: Data subscription status and monthly gigabyte usage
*   `CustServCalls`: Total number of customer service inquiries made
*   `DayMins` & `DayCalls`: Average daytime call volume (minutes and total counts)
*   `MonthlyCharge` & `OverageFee`: Monthly financial billing metrics

### Preprocessing Pipeline:
1.  **Feature/Target Isolation:** Separated target label (`Churn`) from customer attributes.
2.  **Stratified Train-Test Split:** Partitioned data into 80% training and 20% testing sets. Stratification was implemented to ensure the 14.5% churn distribution was perfectly preserved across both splits.
3.  **Feature Scaling:** Standardized numerical fields using `StandardScaler` to prevent feature magnitude bias.

---

## 3. Model Architecture & Evaluation Metrics

We deployed a **Random Forest Classifier** configured with 150 decision estimators. Because the dataset is naturally imbalanced (85.5% loyal vs. 14.5% churned), we embedded `class_weight='balanced'` directly into the algorithm. This forces the model to heavily penalize missing a churned customer, optimizing it for real-world risk detection.

### Classification Performance Results

```text
================================== MODEL PERFORMANCE ==================================
                 precision    recall  f1-score   support

    0 (Stayed)        0.94      0.98      0.96       570
    1 (Churned)       0.81      0.62      0.70        97

      accuracy                            0.92       667
     macro avg        0.87      0.80      0.83       667
  weighted avg        0.92      0.92      0.92       667
```

### Deconstructing the Performance:
*   **92% Overall Accuracy:** The model correctly classifies 92 out of every 100 customers in a completely unseen evaluation set.
*   **81% Churn Precision:** When the model flags a customer as a churn risk, it is mathematically correct **81% of the time**. This ensures marketing budgets are spent exclusively on high-risk users rather than being wasted on safe accounts.
*   **62% Churn Recall (F1-Score: 0.70):** The model actively intercepts **62% of all actual churners** before they exit. An F1-score of 0.70 represents a highly reliable, production-ready benchmark for behavioral churn tasks.

---

## 4. Key Churn Drivers (Data Insights)

The Random Forest architecture calculates an exact mathematical "Importance Score" for every attribute. The top three drivers account for **over 52% of all churn behavior**.

```text
============================ DRIVERS OF CUSTOMER CHURN ============================
         Metric  Importance_Score
        DayMins          0.181057
  CustServCalls          0.177938
  MonthlyCharge          0.164244
ContractRenewal          0.104393
     OverageFee          0.085764
      DataUsage          0.074418
       RoamMins          0.068524
       DayCalls          0.063160
   AccountWeeks          0.060817
       DataPlan          0.019685
```

### Deep-Dive Analysis:
1.  **DayMins (18.1%):** High daytime call volume is the single largest indicator of churn. Heavy communicators rely heavily on network reliability and plan flexibility, making them highly volatile if expectations aren't met.
2.  **CustServCalls (17.7%):** Every call to support is a point of customer friction. Repeated calls indicate unresolved, compounding technical or billing issues.
3.  **MonthlyCharge (16.4%):** Customers are explicitly price-sensitive. Elevated charges without clear utility drive users to aggressively compare market competitors.

---

## 5. Actionable Marketing & Retention Strategies

To translate this machine learning model into direct business revenue, we propose three automated marketing campaigns based on our findings:

*   **Campaign 1: The '3-Call' Proactive Support Trigger**
    *   *Insight:* `CustServCalls` is a major churn catalyst.
    *   *Action:* Establish an automated workflow in the CRM. The moment any account hits **3 or more customer service calls** within a billing cycle, they are automatically placed on a high-priority retention queue. A dedicated support specialist proactively contacts them to resolve their core pain points before they decide to cancel.
*   **Campaign 2: High-Volume Usage Plan Optimization**
    *   *Insight:* High `DayMins` coupled with high `MonthlyCharge` creates an immediate churn risk.
    *   *Action:* Use the model to scan for heavy daytime talkers getting hit with high bills or `OverageFee` spikes. Have marketing automatically message them: *"We noticed your high daytime usage! Let’s move you to our Unlimited High-Volume Plan, which will actually save you $10 a month."* This builds intense brand loyalty and eliminates the price shock.
*   **Campaign 3: Pre-Expiration Lock-In Rewards**
    *   *Insight:* `ContractRenewal` accounts for 10.4% of churn weight.
    *   *Action:* Identify accounts nearing their contract conclusion (`ContractRenewal = 0`). Target these accounts 30 days prior to expiration with automated contract-extension incentives, such as a loyalty credit or a data plan upgrade, locking them in for another 12-month cycle.

---

## Author
Christian Bisrat
