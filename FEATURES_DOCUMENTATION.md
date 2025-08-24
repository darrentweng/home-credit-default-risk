# Home Credit Default Risk - Feature Engineering Documentation

This document describes all the features created in the Home Credit Default Risk prediction project.

## **Feature Overview**

The project creates **80+ engineered features** from 6 different data sources, combining them with the original application features for a comprehensive risk assessment model.

## **Feature Categories**

### **1. Application Features (Original)**
*From `application_train.csv` - Core applicant information*

#### **Demographic Features:**
- `CODE_GENDER` - Gender of the client
- `CNT_CHILDREN` - Number of children
- `CNT_FAM_MEMBERS` - Family size
- `DAYS_BIRTH` - Age in days (negative values)
- `NAME_EDUCATION_TYPE` - Education level
- `NAME_FAMILY_STATUS` - Marital status
- `NAME_INCOME_TYPE` - Income source type
- `OCCUPATION_TYPE` - Job category

#### **Financial Features:**
- `AMT_INCOME_TOTAL` - Total income
- `AMT_CREDIT` - Loan amount requested
- `AMT_ANNUITY` - Loan annuity
- `AMT_GOODS_PRICE` - Price of goods for loan
- `DAYS_EMPLOYED` - Employment duration
- `DAYS_ID_PUBLISH` - Days since ID document issued

#### **External Scores:**
- `EXT_SOURCE_1` - External data source score 1
- `EXT_SOURCE_2` - External data source score 2  
- `EXT_SOURCE_3` - External data source score 3

#### **Housing & Contact Info:**
- `FLAG_OWN_CAR` - Owns a car
- `FLAG_OWN_REALTY` - Owns real estate
- `FLAG_MOBIL` - Provided mobile phone
- `FLAG_EMAIL` - Provided email
- Building information features (`*_AVG`, `*_MODE`, `*_MEDI`)

---

### **2. Bureau Features (Engineered)**
*From `bureau.csv` - Credit Bureau history*

#### **Credit History Volume:**
- `bureau_credit_count` - Total number of previous credits
- `bureau_active_credit_count` - Number of active credits
- `bureau_closed_credit_count` - Number of closed credits

#### **Credit Amounts:**
- `bureau_avg_credit_amount` - Average credit amount
- `bureau_total_debt` - Total current debt
- `bureau_max_overdue_amount` - Maximum overdue amount ever

#### **Credit Behavior:**
- `bureau_credit_utilization_ratio` - Debt to credit limit ratio
- `bureau_avg_days_credit` - Average days since credit application
- `bureau_avg_days_overdue` - Average days overdue
- `bureau_credit_prolong_count` - Number of credit prolongations

#### **Credit Types & Status:**
- Credit type distributions (Cash, Card, Mortgage, etc.)
- Credit status distributions (Active, Closed, Sold, etc.)

---

### **3. Previous Application Features (Engineered)**
*From `previous_application.csv` - Previous Home Credit applications*

#### **Application History:**
- `prev_app_count` - Number of previous applications
- `prev_app_approved_count` - Number of approved applications
- `prev_app_refused_count` - Number of refused applications
- `prev_app_approval_rate` - Approval rate (approved/total)

#### **Previous Loan Amounts:**
- `prev_app_avg_credit` - Average previous credit amount
- `prev_app_avg_annuity` - Average previous annuity
- `prev_app_avg_goods_price` - Average goods price
- `prev_app_max_credit` - Maximum previous credit
- `prev_app_total_credit` - Total previous credit

#### **Application Timing:**
- `prev_app_avg_days_decision` - Average days to decision
- `prev_app_recent_application` - Days since most recent application

#### **Contract Types:**
- Previous contract type distributions
- Payment type preferences
- Channel preferences (online, store, etc.)

---

### **4. Installments Features (Engineered)**
*From `installments_payments.csv` - Payment history*

#### **Payment Behavior:**
- `installments_count` - Total number of installments
- `installments_payment_ratio` - Average payment to installment ratio
- `installments_late_payment_count` - Number of late payments
- `installments_payment_consistency` - Payment consistency score

#### **Payment Amounts:**
- `installments_avg_payment_amount` - Average payment amount
- `installments_avg_instalment_amount` - Average installment amount
- `installments_total_paid` - Total amount paid
- `installments_overpayment_ratio` - Overpayment tendency

#### **Payment Timing:**
- `installments_avg_days_late` - Average days late
- `installments_max_days_late` - Maximum days late
- `installments_on_time_ratio` - On-time payment ratio

#### **Payment Patterns:**
- Early payment frequency
- Payment amount volatility
- Seasonal payment patterns

---

### **5. Credit Card Features (Engineered)**
*From `credit_card_balance.csv` - Credit card usage*

#### **Balance & Utilization:**
- `cc_balance_count` - Number of credit card months
- `cc_avg_balance_utilization` - Average balance utilization
- `cc_max_balance_utilization` - Maximum utilization
- `cc_avg_balance` - Average balance
- `cc_avg_credit_limit` - Average credit limit

#### **Payment Behavior:**
- `cc_payment_behavior_score` - Overall payment behavior score
- `cc_avg_payment_amount` - Average payment amount
- `cc_payment_consistency` - Payment consistency
- `cc_min_payment_ratio` - Minimum payment compliance

#### **Spending Patterns:**
- `cc_total_drawings` - Total cash drawings
- `cc_avg_drawings_atm` - Average ATM withdrawals
- `cc_avg_drawings_pos` - Average POS transactions
- `cc_spending_volatility` - Spending pattern volatility

#### **Delinquency:**
- `cc_avg_dpd` - Average days past due
- `cc_max_dpd` - Maximum days past due
- `cc_dpd_frequency` - Frequency of being past due

#### **Account Management:**
- `cc_active_months` - Number of active months
- Contract status distributions
- Credit limit changes over time