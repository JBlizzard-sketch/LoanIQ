            # File: models/synthetic_data.py

            import numpy as np
            import pandas as pd
            from dataclasses import dataclass, field
            from typing import Dict, List, Optional

            # PATCH START: Kenya-aware SyntheticDataGenerator (configurable)

            # Basic Kenyan name seeds (English first name, Kenyan last name to infer gender)
            FIRST_NAMES_FEMALE = [
                "Mary","Jane","Grace","Ann","Ruth","Lydia","Alice","Mercy","Brenda","Elizabeth",
                "Sarah","Esther","Rachel","Irene","Janet","Agnes","Joy","Naomi","Diana","Cynthia",
            ]
            FIRST_NAMES_MALE = [
                "John","Peter","Michael","David","James","Paul","Samuel","Daniel","Joseph","Stephen",
                "Brian","George","Alex","Allan","Felix","Eric","Kevin","Martin","Francis","Robert",
            ]
            KENYAN_SURNAMES = [
                "Njuguna","Onyango","Odhiambo","Wambui","Mutua","Kiprono","Kiptoo","Mwangi","Otieno","Koech",
                "Ndirangu","Kilonzo","Wekesa","Muthoni","Omondi","Chebet","Cheruiyot","Mutuku","Wanjiru","Achieng",
                "Njeri","Wanyama","Barasa","Gathoni","Kariuki","Kamau","Waweru","Maina","Ouma","Were",
            ]

            # Regions/branches (biased towards Eastern, Rift, Coast, Central, Nairobi)
            REGIONS = {
                "Nairobi": ["CBD","Westlands","Kayole","Kawangware","Kibera","Embakasi","Githurai","Donholm"],
                "Central": ["Thika","Nyeri","Murang'a","Kiambu","Embu","Kerugoya","Karatina","Chuka"],
                "Eastern": ["Meru","Isiolo","Marsabit","Maua","Mwingi","Kitui","Machakos","Makueni","Wote"],
                "Rift": ["Eldoret","Nakuru","Naivasha","Kericho","Bomet","Nandi Hills","Kapsabet","Kitale"],
                "Coast": ["Mombasa","Likoni","Changamwe","Kilifi","Malindi","Lamu","Voi","Mtwapa"],
                "Western": ["Kakamega","Bungoma","Busia","Vihiga"],
                "Nyanza": ["Kisumu","Homa Bay","Migori","Siaya","Kisii","Nyamira"],
                "North Eastern": ["Garissa","Wajir","Mandera"],
            }

            DEFAULT_REGION_WEIGHTS = {
                "Nairobi": 0.22, "Central": 0.18, "Eastern": 0.18, "Rift": 0.20, "Coast": 0.15,
                "Western": 0.03, "Nyanza": 0.03, "North Eastern": 0.01
            }

            DEFAULT_OCCUPATIONS = {
                # skew toward small businesses and informal sector; female-leaning categories included
                "mama_mboga": 0.25,        # ~25% women market traders
                "shop_owner": 0.15,
                "boda_boda": 0.15,
                "salon_spa": 0.08,
                "tailor": 0.07,
                "casual_worker": 0.07,
                "farmer_smallscale": 0.07,
                "teacher": 0.05,
                "civil_servant": 0.04,
                "security_guard": 0.03,
                "student": 0.02,
                "unemployed": 0.02
            }

            DEFAULT_PURPOSES = {
                "working_capital": 0.45,
                "stock_purchase": 0.25,
                "school_fees": 0.10,
                "medical": 0.06,
                "asset_purchase": 0.08,
                "household": 0.06
            }

            @dataclass
            class GeneratorConfig:
                n_samples: int = 5000
                random_state: int = 42
                female_share: float = 0.62                 # skew toward women borrowers
                loan_min: int = 5_000
                loan_max: int = 100_000                    # requirement: 5k–100k, skew small
                loan_skew_sigma: float = 0.55              # larger => more left skew
                term_weeks_choices: List[int] = field(default_factory=lambda: [4,8,12,16,20,24,36,52])
                term_probs: List[float] = field(default_factory=lambda: [0.18,0.20,0.18,0.14,0.10,0.10,0.06,0.04])
                region_weights: Dict[str, float] = field(default_factory=lambda: DEFAULT_REGION_WEIGHTS.copy())
                occupations: Dict[str, float] = field(default_factory=lambda: DEFAULT_OCCUPATIONS.copy())
                purposes: Dict[str, float] = field(default_factory=lambda: DEFAULT_PURPOSES.copy())
                default_rate_target: float = 0.12          # ~12% portfolio default
                use_names: bool = True

            class SyntheticDataGenerator:
                """Configurable, Kenya-aware synthetic generator for LoanIQ."""

                def __init__(self, config: Optional[GeneratorConfig] = None, **kwargs):
                    if config is None:
                        config = GeneratorConfig(**kwargs)
                    self.cfg = config
                    np.random.seed(self.cfg.random_state)

                    # Normalize weights
                    self._norm_inplace(self.cfg.region_weights)
                    self._norm_inplace(self.cfg.occupations)
                    self._norm_inplace(self.cfg.purposes)

                    # Build a 70-branch list across regions (stable order)
                    self.branches = self._build_branches(70)

                @staticmethod
                def _norm_inplace(d: Dict[str, float]):
                    s = float(sum(d.values()))
                    if s <= 0:  # fallback equal
                        k = len(d)
                        for key in d: d[key] = 1.0 / k
                    else:
                        for key in d: d[key] /= s

                def _build_branches(self, n):
                    # allocate branches proportional to region weights
                    regions = list(self.cfg.region_weights.keys())
                    weights = np.array([self.cfg.region_weights[r] for r in regions])
                    weights = weights / weights.sum()
                    alloc = np.random.multinomial(n, weights)
                    branches = []
                    for r, k in zip(regions, alloc):
                        pool = REGIONS[r]
                        for i in range(k):
                            branches.append((r, pool[i % len(pool)]))
                    np.random.shuffle(branches)
                    return branches

                # ---- main API ----
                def to_dataframe(self) -> pd.DataFrame:
                    n = self.cfg.n_samples
                    # Region/branch
                    reg_idx = np.random.choice(len(self.branches), size=n, replace=True)
                    region = [self.branches[i][0] for i in reg_idx]
                    branch = [self.branches[i][1] for i in reg_idx]

                    # Gender & names
                    is_female = np.random.rand(n) < self.cfg.female_share
                    if self.cfg.use_names:
                        first = [
                            np.random.choice(FIRST_NAMES_FEMALE if f else FIRST_NAMES_MALE)
                            for f in is_female
                        ]
                        last = [np.random.choice(KENYAN_SURNAMES) for _ in range(n)]
                        name = [f"{a} {b}" for a, b in zip(first, last)]
                    else:
                        first = [""]*n
                        last = [""]*n
                        name = [""]*n

                    gender = np.where(is_female, "female", "male")

                    # Age 21–65 (slightly younger bias)
                    age = np.clip(np.random.normal(34, 9, n), 21, 65).astype(int)

                    # Dependents (0–6, skew small)
                    dependents = np.random.choice(
                        [0,1,2,3,4,5,6], size=n, p=[0.20,0.27,0.24,0.16,0.08,0.04,0.01]
                    )

                    # Education
                    education = np.random.choice(
                        ["none","primary","secondary","college","university"],
                        size=n, p=[0.05,0.30,0.38,0.20,0.07]
                    )

                    # Occupation
                    occ_keys = list(self.cfg.occupations.keys())
                    occ_probs = [self.cfg.occupations[k] for k in occ_keys]
                    occupation = np.random.choice(occ_keys, size=n, p=occ_probs)

                    # Group / guarantor
                    group_membership = np.random.rand(n) < 0.55
                    guarantor = np.random.rand(n) < 0.62

                    # Income (monthly, KES) — lognormal & occupation-dependent
                    base_income = np.random.lognormal(mean=10.4, sigma=0.45, size=n)  # around 33k mean
                    income = base_income
                    # occupation adjustments
                    adj = {
                        "mama_mboga": 0.9, "shop_owner": 1.15, "boda_boda": 0.95, "salon_spa": 0.95,
                        "tailor": 0.9, "casual_worker": 0.75, "farmer_smallscale": 0.85, "teacher": 1.2,
                        "civil_servant": 1.35, "security_guard": 0.85, "student": 0.6, "unemployed": 0.5
                    }
                    mult = np.array([adj.get(o,1.0) for o in occupation])
                    income = (income * mult).clip(8_000, 180_000).astype(int)

                    # Mobile money activity (transactions/month)
                    mobile_money_txns = np.clip(
                        np.random.normal(55, 25, n) * (income/35_000) ** 0.2, 2, 250
                    ).astype(int)

                    # Past loan history
                    past_loans = np.random.choice([0,1,2,3,4,5], size=n, p=[0.28,0.26,0.20,0.14,0.08,0.04])
                    past_defaults = np.clip(
                        (np.random.rand(n) < 0.18).astype(int) + (past_loans > 3).astype(int) * (np.random.rand(n) < 0.25),
                        0, 3
                    )

                    # Loan attributes
                    # lognormal -> scale to [loan_min, loan_max], skew to small loans
                    raw = np.random.lognormal(mean=np.log(20_000), sigma=self.cfg.loan_skew_sigma, size=n)
                    loan_amount = np.clip(raw, self.cfg.loan_min, self.cfg.loan_max).astype(int)
                    loan_term_weeks = np.random.choice(self.cfg.term_weeks_choices, size=n, p=self.cfg.term_probs)
                    product = np.random.choice(
                        ["Inuka 4 Weeks","Kuza 4 Weeks","Fadhili 4 Weeks","Biashara 12 Weeks","Jijenge 24 Weeks"],
                        size=n, p=[0.28,0.26,0.18,0.18,0.10]
                    )
                    purpose_keys = list(self.cfg.purposes.keys())
                    purpose_probs = [self.cfg.purposes[k] for k in purpose_keys]
                    loan_purpose = np.random.choice(purpose_keys, size=n, p=purpose_probs)
                    loan_type = np.random.choice(["Normal","SME","Emergency"], size=n, p=[0.82,0.12,0.06])

                    # Status / health at origination (ledger)
                    status = np.random.choice(["Active","Pending Branch Approval","Rejected"], size=n, p=[0.78,0.17,0.05])
                    created_date = pd.to_datetime("today").normalize()

                    # --- Default probability model (heuristic for label) ---
                    # base: 12%
                    p = np.full(n, self.cfg.default_rate_target, dtype=float)

                    # risk up with loan size, term, prior defaults; down with income, guarantor, group
                    p += 0.12 * (loan_amount/100_000) ** 0.8
                    p += 0.06 * (loan_term_weeks/52) ** 0.9
                    p += 0.10 * (past_defaults > 0)
                    p += 0.06 * (dependents >= 3)
                    p += 0.04 * (occupation == "unemployed")
                    p += 0.03 * (occupation == "casual_worker")
                    p += 0.02 * (occupation == "boda_boda")
                    p -= 0.08 * (income >= 60_000)
                    p -= 0.05 * guarantor
                    p -= 0.04 * group_membership
                    p -= 0.03 * (mobile_money_txns >= 80)

                    # clamp + noise
                    p = np.clip(p + np.random.normal(0, 0.015, n), 0.01, 0.65)
                    default_flag = (np.random.rand(n) < p).astype(int)
                    loan_health = np.where(default_flag == 1, "Defaulted", "Performing")

                    # Build DataFrame
                    df = pd.DataFrame({
                        "customer_id": np.arange(1, n+1),
                        "name": name,
                        "gender": gender,
                        "age": age,
                        "marital_status": np.random.choice(["single","married","divorced","widowed"], size=n, p=[0.46,0.44,0.06,0.04]),
                        "education_level": education,
                        "occupation": occupation,
                        "household_income": income,
                        "dependents": dependents,
                        "group_membership": np.where(group_membership, "yes", "no"),
                        "guarantor": np.where(guarantor, "yes", "no"),
                        "mobile_money_txns": mobile_money_txns,
                        "past_loans": past_loans,
                        "past_defaults": past_defaults,
                        "region": region,
                        "branch": branch,
                        "product": product,
                        "loan_purpose": loan_purpose,
                        "loan_amount": loan_amount,
                        "loan_term_weeks": loan_term_weeks,
                        "loan_type": loan_type,
                        "status": status,
                        "loan_health": loan_health,
                        "created_date": created_date,
                        "default_flag": default_flag,  # <-- target
                    })

                    return df

            # PATCH END            ["salaried", "self-employed", "unemployed"],
            size=self.n_samples,
            p=[0.6, 0.3, 0.1])

        # Gender
        gender = np.random.choice(["male", "female"],
                                  size=self.n_samples,
                                  p=[0.5, 0.5])

        # Credit score: 300–850, normal-ish distribution
        credit_score = np.random.normal(620, 80,
                                        self.n_samples).clip(300,
                                                             850).astype(int)

        # Default probability (logistic function with noise)
        base_prob = (0.3 * (loan_amount / 80000) + 0.3 *
                     (1 - credit_score / 850) + 0.2 * (1 - income / 120000) +
                     0.2 * (np.random.rand(self.n_samples)))
        defaulted = (base_prob > 0.5).astype(int)

        # Assemble DataFrame
        df = pd.DataFrame({
            "age": age,
            "income": income,
            "loan_amount": loan_amount,
            "loan_term": loan_term,
            "employment_status": employment_status,
            "gender": gender,
            "credit_score": credit_score,
            "defaulted": defaulted,
        })

        return df

    def to_dataframe(self):
        return self.generate()


# PATCH END
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from scipy import stats


class SyntheticDataGenerator:

    def __init__(self):
        self.random_state = 42
        np.random.seed(self.random_state)
        random.seed(self.random_state)

        # Define realistic distributions and correlations
        self.age_dist = {'mean': 40, 'std': 12, 'min': 18, 'max': 80}
        self.income_dist = {'mean': 11.0, 'std': 0.6}  # Log-normal parameters
        self.credit_score_dist = {
            'mean': 650,
            'std': 80,
            'min': 300,
            'max': 850
        }

        # Employment categories with realistic distributions
        self.employment_categories = {
            '< 1 year': 0.15,
            '1-2 years': 0.20,
            '3-5 years': 0.30,
            '5-10 years': 0.25,
            '10+ years': 0.10
        }

        self.home_ownership_categories = {
            'RENT': 0.45,
            'MORTGAGE': 0.35,
            'OWN': 0.15,
            'OTHER': 0.05
        }

        self.loan_purpose_categories = {
            'debt_consolidation': 0.40,
            'credit_card': 0.20,
            'home_improvement': 0.15,
            'major_purchase': 0.10,
            'medical': 0.08,
            'other': 0.07
        }

        # Risk factor weights for calculating default probability
        self.risk_weights = {
            'credit_score': -0.008,  # Lower credit score increases risk
            'debt_to_income_ratio': 3.0,  # Higher DTI increases risk
            'annual_income': -0.00002,  # Higher income decreases risk
            'age': -0.01,  # Older age slightly decreases risk
            'employment_length_numeric':
            -0.1,  # Longer employment decreases risk
            'loan_amount': 0.00001,  # Higher loan amount increases risk
            'intercept': 2.5
        }

    def generate_data(self, config):
        """Generate synthetic loan application data"""
        num_records = config.get('num_records', 1000)
        risk_distribution = config.get('risk_distribution', 'balanced')
        include_defaults = config.get('include_defaults', True)
        add_noise = config.get('add_noise', True)

        # Initialize data structure
        data = []

        for i in range(num_records):
            record = self._generate_single_record(risk_distribution, add_noise)

            # Calculate default probability and default flag
            if include_defaults:
                default_prob = self._calculate_default_probability(record)
                record['default_probability'] = default_prob
                record['default'] = 1 if np.random.random(
                ) < default_prob else 0
            else:
                record['default_probability'] = 0.0
                record['default'] = 0

            # Add application ID
            record[
                'application_id'] = f"APP_{datetime.now().strftime('%Y%m%d')}_{i+1:06d}"
            record['application_date'] = self._generate_random_date()

            data.append(record)

        # Convert to DataFrame
        df = pd.DataFrame(data)

        # Add derived features
        df = self._add_derived_features(df)

        # Apply economic scenario adjustments
        economic_scenario = config.get('economic_scenario', 'normal')
        df = self._apply_economic_scenario(df, economic_scenario)

        return df

    def _generate_single_record(self, risk_distribution, add_noise):
        """Generate a single loan application record"""
        # Adjust distributions based on risk profile
        if risk_distribution == 'conservative':
            # Shift distributions toward lower risk
            age_mean = self.age_dist['mean'] + 5
            credit_mean = self.credit_score_dist['mean'] + 30
            income_mean = self.income_dist['mean'] + 0.2
        elif risk_distribution == 'aggressive':
            # Shift distributions toward higher risk
            age_mean = self.age_dist['mean'] - 3
            credit_mean = self.credit_score_dist['mean'] - 40
            income_mean = self.income_dist['mean'] - 0.3
        else:  # balanced
            age_mean = self.age_dist['mean']
            credit_mean = self.credit_score_dist['mean']
            income_mean = self.income_dist['mean']

        # Generate core demographic features
        age = int(
            np.clip(np.random.normal(age_mean, self.age_dist['std']),
                    self.age_dist['min'], self.age_dist['max']))

        # Generate income (log-normal distribution)
        log_income = np.random.normal(income_mean, self.income_dist['std'])
        annual_income = int(np.exp(log_income))

        # Generate credit score with correlation to income
        income_factor = (log_income - 10.5) * 20  # Correlation factor
        credit_score = int(
            np.clip(
                np.random.normal(credit_mean + income_factor,
                                 self.credit_score_dist['std']),
                self.credit_score_dist['min'], self.credit_score_dist['max']))

        # Generate employment length (categorical)
        employment_length = np.random.choice(
            list(self.employment_categories.keys()),
            p=list(self.employment_categories.values()))

        # Convert employment length to numeric for calculations
        employment_length_numeric = self._employment_to_numeric(
            employment_length)

        # Generate home ownership
        home_ownership = np.random.choice(
            list(self.home_ownership_categories.keys()),
            p=list(self.home_ownership_categories.values()))

        # Generate loan purpose
        loan_purpose = np.random.choice(
            list(self.loan_purpose_categories.keys()),
            p=list(self.loan_purpose_categories.values()))

        # Generate loan amount (correlated with income)
        loan_amount_factor = annual_income * np.random.uniform(0.1, 0.8)
        loan_amount = int(np.clip(loan_amount_factor, 1000, 100000))

        # Generate debt-to-income ratio (with realistic constraints)
        base_dti = np.random.beta(2, 5)  # Skewed toward lower values
        if credit_score < 600:
            base_dti *= 1.5  # Higher DTI for lower credit scores
        debt_to_income_ratio = np.clip(base_dti, 0.01, 0.80)

        # Add noise if requested
        if add_noise:
            credit_score += int(np.random.normal(0, 5))
            annual_income += int(np.random.normal(0, annual_income * 0.05))
            debt_to_income_ratio += np.random.normal(0, 0.02)

            # Ensure values stay within bounds
            credit_score = np.clip(credit_score, 300, 850)
            annual_income = max(10000, annual_income)
            debt_to_income_ratio = np.clip(debt_to_income_ratio, 0.01, 0.80)

        # Calculate monthly income and expenses
        monthly_income = annual_income / 12
        monthly_debt_payment = monthly_income * debt_to_income_ratio

        return {
            'age': age,
            'annual_income': annual_income,
            'monthly_income': monthly_income,
            'credit_score': credit_score,
            'employment_length': employment_length,
            'employment_length_numeric': employment_length_numeric,
            'home_ownership': home_ownership,
            'loan_purpose': loan_purpose,
            'loan_amount': loan_amount,
            'debt_to_income_ratio': round(debt_to_income_ratio, 4),
            'monthly_debt_payment': round(monthly_debt_payment, 2)
        }

    def _employment_to_numeric(self, employment_length):
        """Convert employment length to numeric years"""
        mapping = {
            '< 1 year': 0.5,
            '1-2 years': 1.5,
            '3-5 years': 4.0,
            '5-10 years': 7.5,
            '10+ years': 12.0
        }
        return mapping.get(employment_length, 0)

    def _calculate_default_probability(self, record):
        """Calculate realistic default probability based on features"""
        # Linear combination of risk factors
        risk_score = self.risk_weights['intercept']

        for feature, weight in self.risk_weights.items():
            if feature != 'intercept' and feature in record:
                risk_score += record[feature] * weight

        # Convert to probability using logistic function
        default_prob = 1 / (1 + np.exp(-risk_score))

        # Add some randomness but keep realistic
        default_prob += np.random.normal(0, 0.05)
        default_prob = np.clip(default_prob, 0.01, 0.95)

        return round(default_prob, 4)

    def _generate_random_date(self):
        """Generate random application date within the last 2 years"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)

        time_between = end_date - start_date
        days_between = time_between.days
        random_days = random.randrange(days_between)

        return start_date + timedelta(days=random_days)

    def _add_derived_features(self, df):
        """Add derived features to the dataset"""
        # Payment-to-income ratio
        df['payment_to_income_ratio'] = (df['loan_amount'] * 0.05) / df[
            'annual_income']  # Assuming 5% interest

        # Credit utilization estimate (synthetic)
        df['estimated_credit_utilization'] = df[
            'debt_to_income_ratio'] * np.random.uniform(0.3, 0.8, len(df))

        # Age groups
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '55+'])

        # Income quartiles
        df['income_quartile'] = pd.qcut(df['annual_income'],
                                        q=4,
                                        labels=['Q1', 'Q2', 'Q3', 'Q4'])

        # Credit score bands
        df['credit_score_band'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])

        # Loan amount categories
        df['loan_size_category'] = pd.cut(
            df['loan_amount'],
            bins=[0, 5000, 15000, 35000, 100000],
            labels=['Small', 'Medium', 'Large', 'Very Large'])

        # Risk indicators
        df['high_dti_flag'] = (df['debt_to_income_ratio'] > 0.4).astype(int)
        df['low_credit_flag'] = (df['credit_score'] < 600).astype(int)
        df['high_loan_amount_flag'] = (df['loan_amount'] > 50000).astype(int)

        return df

    def _apply_economic_scenario(self, df, scenario):
        """Apply economic scenario adjustments"""
        if scenario == 'recession':
            # Increase default rates across the board
            df['default_probability'] = df['default_probability'] * 1.5
            df['default'] = (np.random.random(len(df))
                             < df['default_probability']).astype(int)

            # Adjust credit scores downward
            df['credit_score'] = np.clip(df['credit_score'] - 20, 300, 850)

        elif scenario == 'boom':
            # Decrease default rates
            df['default_probability'] = df['default_probability'] * 0.7
            df['default'] = (np.random.random(len(df))
                             < df['default_probability']).astype(int)

            # Adjust incomes upward
            df['annual_income'] = df['annual_income'] * 1.1

        elif scenario == 'high_inflation':
            # Increase loan amounts (people need more money)
            df['loan_amount'] = df['loan_amount'] * 1.2

            # Increase debt-to-income ratios
            df['debt_to_income_ratio'] = np.clip(
                df['debt_to_income_ratio'] * 1.1, 0, 0.8)

        # Ensure default probability stays within bounds
        df['default_probability'] = np.clip(df['default_probability'], 0.01,
                                            0.95)

        return df

    def generate_edge_cases(self, num_cases=100):
        """Generate edge cases for testing"""
        edge_cases = []

        for i in range(num_cases):
            case_type = random.choice([
                'extreme_high_risk', 'extreme_low_risk', 'boundary_values',
                'missing_data'
            ])

            if case_type == 'extreme_high_risk':
                record = {
                    'age': random.randint(18, 25),
                    'annual_income': random.randint(15000, 25000),
                    'credit_score': random.randint(300, 500),
                    'debt_to_income_ratio': random.uniform(0.6, 0.8),
                    'employment_length': '< 1 year',
                    'home_ownership': 'RENT',
                    'loan_purpose': 'debt_consolidation',
                    'loan_amount': random.randint(25000, 50000)
                }

            elif case_type == 'extreme_low_risk':
                record = {
                    'age': random.randint(35, 55),
                    'annual_income': random.randint(150000, 300000),
                    'credit_score': random.randint(750, 850),
                    'debt_to_income_ratio': random.uniform(0.05, 0.15),
                    'employment_length': '10+ years',
                    'home_ownership': 'OWN',
                    'loan_purpose': 'home_improvement',
                    'loan_amount': random.randint(5000, 15000)
                }

            elif case_type == 'boundary_values':
                record = {
                    'age':
                    random.choice([18, 80]),
                    'annual_income':
                    random.choice([10000, 500000]),
                    'credit_score':
                    random.choice([300, 850]),
                    'debt_to_income_ratio':
                    random.choice([0.01, 0.79]),
                    'employment_length':
                    random.choice(['< 1 year', '10+ years']),
                    'home_ownership':
                    random.choice(list(self.home_ownership_categories.keys())),
                    'loan_purpose':
                    random.choice(list(self.loan_purpose_categories.keys())),
                    'loan_amount':
                    random.choice([1000, 100000])
                }

            else:  # missing_data
                record = {
                    'age':
                    random.randint(25, 65) if random.random() > 0.3 else None,
                    'annual_income':
                    random.randint(30000, 100000)
                    if random.random() > 0.2 else None,
                    'credit_score':
                    random.randint(500, 750)
                    if random.random() > 0.1 else None,
                    'debt_to_income_ratio':
                    random.uniform(0.1, 0.5)
                    if random.random() > 0.4 else None,
                    'employment_length':
                    random.choice(list(self.employment_categories.keys()))
                    if random.random() > 0.3 else None,
                    'home_ownership':
                    random.choice(list(self.home_ownership_categories.keys()))
                    if random.random() > 0.2 else None,
                    'loan_purpose':
                    random.choice(list(self.loan_purpose_categories.keys()))
                    if random.random() > 0.1 else None,
                    'loan_amount':
                    random.randint(5000, 50000)
                    if random.random() > 0.2 else None
                }

            # Add metadata
            record['case_type'] = case_type
            record['application_id'] = f"EDGE_{i+1:04d}"
            record['application_date'] = self._generate_random_date()

            edge_cases.append(record)

        return pd.DataFrame(edge_cases)

    def generate_time_series_data(self, days=365):
        """Generate time series data for trend analysis"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        time_series_data = []

        for i, date in enumerate(dates):
            # Vary the number of applications per day
            daily_applications = max(
                1,
                int(np.random.poisson(15) + np.sin(i * 2 * np.pi / 365) * 5))

            for j in range(daily_applications):
                # Generate application with some seasonal effects
                seasonal_factor = 1 + 0.2 * np.sin(
                    i * 2 * np.pi / 365)  # Annual seasonality
                weekly_factor = 1 + 0.1 * np.sin(
                    i * 2 * np.pi / 7)  # Weekly seasonality

                config = {
                    'num_records': 1,
                    'risk_distribution': 'balanced',
                    'include_defaults': True,
                    'add_noise': True
                }

                record = self._generate_single_record('balanced', True)

                # Apply seasonal adjustments
                record['annual_income'] = int(record['annual_income'] *
                                              seasonal_factor)
                record['loan_amount'] = int(record['loan_amount'] *
                                            weekly_factor)

                # Calculate default probability and flag
                default_prob = self._calculate_default_probability(record)
                record['default_probability'] = default_prob
                record['default'] = 1 if np.random.random(
                ) < default_prob else 0

                record[
                    'application_id'] = f"TS_{date.strftime('%Y%m%d')}_{j+1:03d}"
                record['application_date'] = date

                time_series_data.append(record)

        df = pd.DataFrame(time_series_data)
        df = self._add_derived_features(df)

        return df

    def validate_generated_data(self, df):
        """Validate the quality of generated synthetic data"""
        validation_results = {
            'total_records': len(df),
            'missing_values': df.isnull().sum().sum(),
            'duplicate_records': df.duplicated().sum(),
            'data_quality_issues': []
        }

        # Check for realistic value ranges
        if df['age'].min() < 18 or df['age'].max() > 100:
            validation_results['data_quality_issues'].append(
                "Age values outside realistic range")

        if df['credit_score'].min() < 300 or df['credit_score'].max() > 850:
            validation_results['data_quality_issues'].append(
                "Credit score values outside valid range")

        if df['debt_to_income_ratio'].min(
        ) < 0 or df['debt_to_income_ratio'].max() > 1:
            validation_results['data_quality_issues'].append(
                "DTI ratio values outside valid range")

        if df['annual_income'].min() < 0:
            validation_results['data_quality_issues'].append(
                "Negative income values found")

        # Check distributions
        default_rate = df['default'].mean()
        if default_rate < 0.02 or default_rate > 0.5:
            validation_results['data_quality_issues'].append(
                f"Default rate ({default_rate:.3f}) seems unrealistic")

        # Check correlations
        income_credit_corr = df['annual_income'].corr(df['credit_score'])
        if income_credit_corr < 0.1:
            validation_results['data_quality_issues'].append(
                "Income-credit score correlation is too weak")

        validation_results['default_rate'] = default_rate
        validation_results['income_credit_correlation'] = income_credit_corr
        validation_results['is_valid'] = len(
            validation_results['data_quality_issues']) == 0

        return validation_results
