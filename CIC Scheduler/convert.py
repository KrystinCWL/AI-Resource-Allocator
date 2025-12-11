import pandas as pd
import numpy as np
import random
import re
import os
import google.genai as genai

# --- Cell 2: Colab Setup (Commented Out for .py) and Constants ---
# The following lines are for Google Colab environment setup and are commented out.
# If running locally, ensure your files are accessible at the specified paths.
# from google.colab import drive
# drive.mount('/content/drive')

# Define the path to your main XLSX file.
XLSX_PATH = "/content/drive/MyDrive/CIC Algorithm/user_db_V2.xlsx"
OUTPUT_DIR = "/content/drive/MyDrive/CIC Algorithm/Schedule_Results"

# Maximum number of sections per faculty
MAX_LOAD = 3
# Assume FLEXIBLE_MAX_LOAD_CAP is already defined, for example 5
FLEXIBLE_MAX_LOAD_CAP = 5

# --- Cell 3: Load Data from XLSX ---
# --- Sheet Names to DataFrame Variable Mapping ---
# Dictionary mapping desired DataFrame variable names to the sheet names in the Excel file.
SHEET_MAP = {
    "semester_df": "CIC_Semester",
    "course_df": "CIC_Course",
    "offer_df": "CIC_CourseOffering",
    "faculty_df": "CIC_FacultyProfile",
    "preference_df": "CIC_FacultyPreference",
}

# Dictionary to hold the loaded DataFrames
dataframes = {}

print(f"--- Starting to read data from: {XLSX_PATH} ---")

# Load each required sheet into a separate DataFrame
for df_var_name, sheet_name in SHEET_MAP.items():
    try:
        # Read the Excel file using the specified sheet name
        df = pd.read_excel(XLSX_PATH, sheet_name=sheet_name)

        dataframes[df_var_name] = df
        print(f"✅ Successfully loaded '{sheet_name}' into DataFrame: {df_var_name} (Shape: {df.shape})")

    except Exception as e:
        print(f"❌ ERROR loading sheet '{sheet_name}': {e}")

# Assign DataFrames to the specific variables requested
semester_df = dataframes.get("semester_df")
course_df = dataframes.get("course_df")
offer_df = dataframes.get("offer_df")
faculty_df = dataframes.get("faculty_df")
preference_df = dataframes.get("preference_df")

# --- Cell 4: Install Dependencies (Environment Setup) ---
# !pip install sqlalchemy pymysql

# --- Cells 5-8: MySQL Load (All lines are already commented out in the source) ---
# from sqlalchemy import create_engine
# import pymysql

# # ----------------------------------------------------
# # 步驟 1: 配置 MySQL 連線資訊
# # ----------------------------------------------------
# # 請替換成您 MySQL Workbench 中使用的連線細節
# # DB_CONFIG = {
# # 'host': 'localhost', # 範例: 'localhost' 或 '127.0.0.1'
# # 'port': 3306,                     # 範例: 3306 (預設端口)
# # 'user': 'root',
# # 'password': '123456',
# # 'database': 'user_management',
# # }

# # from sqlalchemy import create_engine
# # import pymysql

# # ----------------------------------------------------
# # 步驟 1: 配置 MySQL 連線資訊
# # ----------------------------------------------------
# # 請替換成您 MySQL Workbench 中使用的連線細節
# # DB_CONFIG = {
# # 'host': 'localhost', # 範例: 'localhost' 或 '127.0.0.1'
# # 'port': 3306,                     # 範例: 3306 (預設端口)
# # 'user': 'flaskuser',
# # 'password': 'FlaskUser123!',
# # 'database': 'user_management',
# # }

# # ----------------------------------------------------
# # 步驟 2: 建立資料庫連線 Engine
# # ----------------------------------------------------
# # 構建資料庫連線 URL: 'mysql+pymysql://user:password@host:port/database'
# # DATABASE_URL = (
# # f"mysql+pymysql://{DB_CONFIG['user']}:"
# # f"{DB_CONFIG['password']}@"
# # f"{DB_CONFIG['host']}:"
# # f"{DB_CONFIG['port']}/"
# # f"{DB_CONFIG['database']}"
# # )

# # 創建 SQLAlchemy Engine
# # try:
# # #     engine = create_engine(DATABASE_URL)
# # #     print("✅ MySQL 資料庫連線 Engine 建立成功。")
# # # except Exception as e:
# # #     print(f"❌ 資料庫連線 Engine 建立失敗: {e}")
# # #     raise

# # ----------------------------------------------------
# # # 步驟 3: 定義表格名稱並載入 DataFrames
# # # ----------------------------------------------------
# # # 根據您在 Excel 中的工作表名稱，假設 MySQL 資料庫中的表格名稱與之相同。
# # # TABLE_NAMES_MAP = {
# # # "semester_df": "CIC_Semester",
# # # "course_df": "CIC_Course",
# # # "offer_df": "CIC_CourseOffering",
# # # "faculty_df": "CIC_FacultyProfile",
# # # "preference_df": "CIC_FacultyPreference",
# # # }

# # # 儲存載入的 DataFrames
# # # dataframes = {}

# # # print("\n--- 開始從 MySQL 載入 DataFrames ---")

# # # for df_var_name, table_name in TABLE_NAMES_MAP.items():
# # #     try:
# # #         # 使用 pd.read_sql_query 執行 SELECT * 查詢，並直接載入為 DataFrame
# # #         sql_query = f"SELECT * FROM {table_name};"
# # #         df = pd.read_sql_query(sql_query, engine)

# # #         # 將 DataFrame 存入 dataframes 字典中，使用您期望的變數名作為 key
# # #         dataframes[df_var_name] = df
# # #         print(f"✅ 成功載入表格 '{table_name}' 到 '{df_var_name}' (Shape: {df.shape})")

# # #     except Exception as e:
# # #         print(f"❌ ERROR 載入表格 '{table_name}' 失敗: {e}")
# # #         # 如果載入失敗，應停止並拋出錯誤
# # #         # raise

# # # ----------------------------------------------------
# # # # 步驟 4: 賦值給您的最終變數
# # # ----------------------------------------------------

# # # # semester_df = dataframes.get("semester_df")
# # # # course_df = dataframes.get("course_df")
# # # # offer_df = dataframes.get("offer_df")
# # # # faculty_df = dataframes.get("faculty_df")
# # # # preference_df = dataframes.get("preference_df")

# # # print("\n🎉 所有 DataFrames 成功從 MySQL 載入並賦值完成！")

# # # ----------------------------------------------------
# # # # 步驟 5: 關閉 Engine 連線
# # # ----------------------------------------------------
# # # # engine.dispose()

# --- Cell 9: Data Preparation ---
# --- 1. Optimized Data Preparation: Expand, Merge Category, and Sort ---
print("--- 1. Optimized Data Preparation ---")

# 1.1: Determine Active Semester
try:
    ACTIVE_SEMESTER_ID = semester_df[semester_df['semesterIsActive'] == 'T']['semesterId'].iloc[0]
except IndexError: # Use IndexError as iloc[0] raises this when no rows match
    print("ERROR: Could not find active semester.")

# 1.2: Expand Sections
active_offerings_df = offer_df[offer_df['semesterId'] == ACTIVE_SEMESTER_ID].copy()
list_of_dfs = []
for _, row in active_offerings_df.iterrows():
    n = int(row['sectionQuantity'])
    temp_df = pd.DataFrame([row] * n)
    section_numbers = range(1, n + 1)
    temp_df['sectionId'] = [f"S{i}" for i in section_numbers]
    temp_df['total_sections'] = temp_df['sectionQuantity']
    list_of_dfs.append(temp_df)

df_sections_to_assign = pd.concat(list_of_dfs, ignore_index=True)
df_sections_to_assign = df_sections_to_assign[['offeringId', 'courseId', 'semesterId', 'sectionId', 'total_sections']]

# 1.3: Merge courseCategory
course_category_map = course_df[['courseId', 'courseCategory']]
df_sections_to_assign = pd.merge(
    df_sections_to_assign,
    course_category_map,
    on='courseId',
    how='left'
)

# Initialize assignment column
df_sections_to_assign['assignedFacultyId'] = 'NOT_ASSIGNED'

# 1.4: Define Priority Sorting (Core before Elective, Large Courses first)
# Create a numerical column for sorting: Core=0, Elective=1 (Core comes first)
df_sections_to_assign['category_rank'] = df_sections_to_assign['courseCategory'].apply(
    lambda x: 0 if x == 'Core' else 1
)

# Final Sorting: Category (ASC) -> Total Sections (DESC) -> Offering ID (ASC for stability)
df_sections_to_assign = df_sections_to_assign.sort_values(
    by=['category_rank', 'total_sections', 'offeringId'],
    ascending=[True, False, True],
    kind='mergesort' # Maintain stability for tie-breaking
).reset_index(drop=True)

print(f"✅ Final sections DataFrame prepared and sorted for assignment: {len(df_sections_to_assign)} sections.")
print("Sections are prioritized: Core (large first) -> Elective (large first).")

# --- Cell 10: Display Data (Optional) ---
# display all rows of df_sections_to_assign
pd.set_option('display.max_rows', None)
# print(df_sections_to_assign) # Uncomment this to print the DataFrame

# --- Cell 11: Prepare Faculty Indicator DF and Scoring Functions ---
# --- 2. Prepare Faculty Indicator DF and Scoring Functions ---
print("\n--- 2. Prepare Faculty Indicator DF and Scoring Functions ---\n")

# --- 2.1 Prepare indicator_df: Faculty x Course combinations ---
# Create a list of all unique faculty IDs
unique_faculty_ids = faculty_df['facultyUserId'].unique()

# Create a list of all unique courses offered in the current semester
unique_course_ids = df_sections_to_assign['courseId'].unique()

# Generate a DataFrame of all possible Faculty-Course combinations
# This DataFrame will be the basis for scoring
# facultyUserId, courseId
indicator_df = pd.MultiIndex.from_product(
    [unique_faculty_ids, unique_course_ids],
    names=['facultyUserId', 'courseId']
).to_frame(index=False)


# --- 2.2 Merge Preference and Experience Scores ---

# Merge Priority Rank (Preference)
# Left join on the indicator_df to preserve all combinations. Non-matches will be NaN.
# 'priorityRank' from preference_df is the desired rank (1=highest, 5=lowest)
indicator_df = pd.merge(
    indicator_df,
    preference_df[['facultyUserId', 'courseId', 'priorityRank']],
    on=['facultyUserId', 'courseId'],
    how='left'
)

# Faculty Experience DataFrame
faculty_exp_df = faculty_df[['facultyUserId', 'facultyExperience']].copy()

# Preference Score
max_rank = indicator_df['priorityRank'].max() if not indicator_df.empty else 5
indicator_df['preference_score'] = (max_rank + 1) - indicator_df['priorityRank']

# Experience Score
indicator_df = pd.merge(indicator_df, faculty_exp_df, on='facultyUserId', how='left')
def calculate_experience_score(row):
    course_id = row['courseId']
    experience_str = str(row['facultyExperience'])
    if pd.isna(experience_str) or experience_str == 'NULL':
        return 0
    pattern = r'\b' + re.escape(course_id) + r'\b'
    return 1 if re.search(pattern, experience_str) else 0

indicator_df['experience_score'] = indicator_df.apply(calculate_experience_score, axis=1)


# --- 2.3 Calculate Total Score and Final Cleanup ---
# Total Score: Preference Score (Higher is better) + Experience Score (1 if taught, 0 if not)
# If preference_score is NaN (no preference stated), treat it as 0 for total score calculation.
indicator_df['preference_score'] = indicator_df['preference_score'].fillna(0)
indicator_df['total_score'] = indicator_df['preference_score'] + indicator_df['experience_score']

# Final cleanup: drop temporary columns
indicator_df = indicator_df.drop(columns=['facultyExperience'])

# --- Define Penalty Rank for KPIs ---
# This is the "bad" rank assigned to a faculty member teaching a course they did not list a preference for.
PENALTY_RANK = max_rank + 2 # E.g., if max_rank=5, penalty=7

print(f"✅ Faculty Indicator DataFrame (all faculty-course combos) created: {len(indicator_df)} rows.")

# --- Cell 12: Helper Function: get_best_candidate_df ---
# Helper function to select the best candidates based on score
def get_best_candidate_df(course_id, df_indicator_in):
    """
    Filters the indicator DataFrame for a given course and sorts candidates by score.
    Returns: DataFrame sorted by total_score (DESC), then preference_score (DESC).
    """
    candidates = df_indicator_in[df_indicator_in['courseId'] == course_id].copy()
    # Sort by total_score (desc) and preference_score (desc)
    candidates = candidates.sort_values(
        by=['total_score', 'preference_score'],
        ascending=[False, False]
    ).reset_index(drop=True)
    return candidates

# --- Cell 13: Assignment Function: execute_assignment_method_1_2 ---
def execute_assignment_method_1_2(df_sections_in, df_indicator_in, initial_max_load, PENALTY_RANK, method_type):
    """執行分配，Method 1: 嚴格 Max Load, 嚴格 Preference > 0。"""

    # 使用複本以確保原始數據不被修改
    df_sections = df_sections_in.copy()
    df_indicator = df_indicator_in.copy()

    unique_faculty = df_indicator['facultyUserId'].unique()
    faculty_load = {uid: 0 for uid in unique_faculty}
    current_max_load = initial_max_load

    sections_to_process = df_sections[df_sections['assignedFacultyId'] == 'NOT_ASSIGNED'].index

    for i in sections_to_process:
        section = df_sections.loc[i]
        course_id = section['courseId']

        candidates_df = get_best_candidate_df(course_id, df_indicator)
        assignedFacultyId = 'NOT_ASSIGNED'

        for _, candidate in candidates_df.iterrows():
            candidate_id = candidate['facultyUserId']
            pref_score = candidate['preference_score']

            # 約束檢查 1：Max Load (嚴格 < MAX_LOAD)
            if faculty_load[candidate_id] >= current_max_load:
                continue

            # 約束檢查 2：Method 1 (Preference Score 限制: 必須大於 0)
            if method_type == 1 and pref_score == 0:
                continue

            # 找到最優且符合約束的教師
            assignedFacultyId = candidate_id
            break # 找到第一個符合條件的就分配

        # 執行分配動作
        if assignedFacultyId != 'NOT_ASSIGNED':
            df_sections.loc[i, 'assignedFacultyId'] = assignedFacultyId
            # faculty_load['sections_assigned'] add 1 after assigned to the faculty_load[assignedFacultyId]
            faculty_load[assignedFacultyId] += 1

    # 最終結果彙整
    total_sections = len(df_sections_in)
    return df_sections, faculty_load

# --- Cell 14: Assignment Function: execute_assignment_method_3_4 ---
def execute_assignment_method_3_4(df_sections_in, df_indicator_in, faculty_load_in, initial_max_load, current_max_load, method_type):
    """
    執行分配，Method 3/4: 寬鬆 Max Load, 無 Preference Score 限制。
    這是 Method 3/4 的核心分配邏輯。

    Parameters:
        df_sections_in (pd.DataFrame): 包含所有 Section 的 DataFrame。
        df_indicator_in (pd.DataFrame): Faculty-Course-Score Indicator DataFrame。
        faculty_load_in (dict): 從上一階段傳入的當前教師負載。
        initial_max_load (int): 初始的最大負載 (MAX_LOAD)。
        current_max_load (int): 當前分配階段的最大負載 (例如 FLEXIBLE_MAX_LOAD_CAP)。
        method_type (int): 3 或 4。
    """
    df_sections = df_sections_in.copy()
    faculty_load = faculty_load_in.copy()
    assigned_in_phase = 0

    # 1. 篩選未分配的 Sections
    unassigned_sections = df_sections[df_sections['assignedFacultyId'] == 'NOT_ASSIGNED']
    unassigned_core = unassigned_sections[unassigned_sections['courseCategory'] == 'Core']
    unassigned_elective = unassigned_sections[unassigned_sections['courseCategory'] == 'Elective']

    # 2. 確定本階段要分配的課程範圍
    if method_type == 3:
        # Method 3: 僅分配未分配的 Core 課程 (因為 Elective 課程在 Method 3 中被視為非必需分配)
        sections_to_assign_in_phase = unassigned_core
    else: # method_type == 4
        # Method 4: 分配所有未分配的課程 (Core + Elective)
        sections_to_assign_in_phase = unassigned_sections

    sections_to_process_index = sections_to_assign_in_phase.index

    # 4. 對列表中的 Sections 進行分配嘗試
    for i in sections_to_process_index:
        section = df_sections.loc[i]
        course_id = section['courseId']

        candidates_df = get_best_candidate_df(course_id, df_indicator_in)
        assignedFacultyId = 'NOT_ASSIGNED'

        for _, candidate in candidates_df.iterrows():
            candidate_id = str(candidate['facultyUserId'])

            # 約束檢查 1：Max Load (嚴格 < current_max_load)
            if faculty_load.get(candidate_id, 0) >= current_max_load:
                continue

            # Method 3/4: 無 Preference Score 限制

            # 找到最優且符合約束的教師
            assignedFacultyId = candidate_id
            break

        # 執行分配動作
        if assignedFacultyId != 'NOT_ASSIGNED':
            df_sections.loc[i, 'assignedFacultyId'] = assignedFacultyId
            faculty_load[assignedFacultyId] = faculty_load.get(assignedFacultyId, 0) + 1
            assigned_in_phase += 1

    # 輸出進度
    unassigned_total_after_phase = len(df_sections[df_sections['assignedFacultyId'] == 'NOT_ASSIGNED'])
    print(f"  -> Method {method_type} Phase assigned {assigned_in_phase} sections. Remaining unassigned: {unassigned_total_after_phase}")

    return df_sections, faculty_load

# --- Cell 15: KPI Calculation Function: calculate_kpis_and_summary ---
def calculate_kpis_and_summary(df_output_in, df_indicator_in, initial_max_load, PENALTY_RANK, method_name):
    """
    Calculate KPIs and format results into DataFrames.
    """
    df_output = df_output_in.copy()
    df_indicator = df_indicator_in.copy()

    # --- 1. Prepare Data ---
    total_sections = len(df_output)
    total_core = len(df_output[df_output['courseCategory'] == 'Core'])
    total_elective = len(df_output[df_output['courseCategory'] == 'Elective'])

    # Merge assignment results with indicator info to get preference_score and priorityRank
    # Use a left join to ensure unassigned sections remain (with 'NOT_ASSIGNED' in facultyUserId)
    df_output['facultyUserId'] = df_output['assignedFacultyId'].apply(lambda x: x if x != 'NOT_ASSIGNED' else np.nan)

    # Perform a precise merge on the assigned rows
    assigned_df = df_output[df_output['assignedFacultyId'] != 'NOT_ASSIGNED'].copy()
    assigned_df['facultyUserId'] = assigned_df['assignedFacultyId'] # Already done, but for clarity

    # Perform the merge only on the assigned data
    assigned_df = pd.merge(
        assigned_df.drop(columns=['priorityRank', 'preference_score', 'total_score'], errors='ignore'),
        df_indicator[['facultyUserId', 'courseId', 'priorityRank', 'preference_score', 'total_score']],
        on=['facultyUserId', 'courseId'],
        how='left'
    )

    # Re-combine the assigned data with the unassigned data (if any)
    unassigned_sections = df_output[df_output['assignedFacultyId'] == 'NOT_ASSIGNED']
    df_output = pd.concat([assigned_df, unassigned_sections.drop(columns=['facultyUserId'], errors='ignore')]).reset_index(drop=True)

    # --- 2. Calculate Load Summary ---
    load_counts = assigned_df['assignedFacultyId'].value_counts()
    load_summary = pd.DataFrame(load_counts).reset_index()
    load_summary.columns = ['facultyUserId', 'sections_assigned']
    load_summary_sorted = load_summary.sort_values(by='sections_assigned', ascending=False).reset_index(drop=True)

    # --- 3. Calculate Core KPIs ---
    total_assigned = len(assigned_df)

    # Assigned rates
    total_assigned_rate = total_assigned / total_sections
    assigned_core = len(assigned_df[assigned_df['courseCategory'] == 'Core'])
    assigned_elective = len(assigned_df[assigned_df['courseCategory'] == 'Elective'])
    core_assigned_rate = assigned_core / total_core if total_core > 0 else 0
    elective_assigned_rate = assigned_elective / total_elective if total_elective > 0 else 0


    # --- KPI 1: Average preference rank (Avg Pref Rank) ---
    # Only include assignments where the faculty explicitly expressed a preference (priorityRank > 0)
    assigned_with_pref = assigned_df[assigned_df['priorityRank'] > 0]
    assigned_with_pref_count = len(assigned_with_pref)

    avg_pref_rank = assigned_with_pref['priorityRank'].mean() if assigned_with_pref_count > 0 else 0

    # --- KPI 2: Overall Satisfaction Score ---
    # Give assignments without a stated preference (NaNs/0s) a penalty score PENALTY_RANK
    # Note: since unassigned sections use assignedFacultyId = 'NOT_ASSIGNED',
    # assigned_df['priorityRank'] here should be NaN or an integer > 0.

    # Compute satisfaction by treating NaN as the penalty
    assigned_df['SatisfactionRank'] = assigned_df['priorityRank'].fillna(PENALTY_RANK).astype(int)

    overall_satisfaction_score = assigned_df['SatisfactionRank'].mean() if total_assigned > 0 else 0


    # --- KPI 3: No Preference Rate ---
    # Count assignments where priorityRank is NaN
    no_pref_assigned_count = assigned_df['priorityRank'].isna().sum()
    no_preference_rate = no_pref_assigned_count / total_assigned if total_assigned > 0 else 0

    # ----------------------------------------------------
    # Remaining KPI calculations and output (same as before with slight tweaks)
    # ----------------------------------------------------

    total_unassigned = total_sections - total_assigned

    # Load Distribution KPI
    max_load_achieved = load_summary['sections_assigned'].max() if not load_summary.empty else 0
    load_std_dev = load_summary['sections_assigned'].std() if len(load_summary) > 1 else 0

    # Format percentages
    total_assigned_rate = f"{total_assigned_rate:.1%}"
    core_assigned_rate = f"{core_assigned_rate:.1%}"
    elective_assigned_rate = f"{elective_assigned_rate:.1%}"
    no_preference_rate = f"{no_preference_rate:.1%}"

    kpi_dict = {
        'Method': method_name,
        'Total Assigned Rate': total_assigned_rate,
        'Core Assigned Rate': core_assigned_rate,
        'Elective Assigned Rate': elective_assigned_rate,
        'Avg Pref Rank': avg_pref_rank, # New: only ranks > 0
        'Overall Satisfaction Score': overall_satisfaction_score, # New: with penalty score
        'No Preference Rate': no_preference_rate,
        'Max Load Achieved': max_load_achieved,
        'Load Std Dev': load_std_dev,
        'Total Sections': total_sections,
        'Total Assigned': total_assigned
    }

    return df_output, load_summary_sorted, unassigned_sections, kpi_dict

# --- Cell 16: Display Faculty Indicator DF (Optional) ---
# display all rows of df_faculty_indicator
# print(df_faculty_indicator) # Uncomment this to print the DataFrame

# --- Cell 17: Main Execution Block (Methods 1-4) ---
# --- Main Execution Block: Run all four methods ---

kpi_results = {}

# --- Method 1: Strict Max Load (MAX_LOAD=3), Strict Preference (Pref > 0) ---
print("\n[Method 1: Strict Max Load (3), Strict Preference (Pref > 0)]")
df_m1, load_m1 = execute_assignment_method_1_2(df_sections_to_assign, indicator_df, MAX_LOAD, PENALTY_RANK, method_type=1)
_, load_summary_m1, unassigned_m1, kpi_m1 = calculate_kpis_and_summary(df_m1, indicator_df, MAX_LOAD, PENALTY_RANK, 'Method 1 (Strict, Pref > 0)')
kpi_results['Method 1'] = kpi_m1

# --- Method 2: Strict Max Load (MAX_LOAD=3), No Preference Restriction ---
print("\n[Method 2: Strict Max Load (3), No Preference Restriction]")
df_m2, load_m2 = execute_assignment_method_1_2(df_sections_to_assign, indicator_df, MAX_LOAD, PENALTY_RANK, method_type=2)
_, load_summary_m2, unassigned_m2, kpi_m2 = calculate_kpis_and_summary(df_m2, indicator_df, MAX_LOAD, PENALTY_RANK, 'Method 2 (Strict, No Pref)')
kpi_results['Method 2'] = kpi_m2

# --- Method 3: Flexible Max Load (CAP=FLEXIBLE_MAX_LOAD_CAP), Core Only Assignment ---
print("\n[Method 3: Flexible Max Load (Cap=5), Core Only Assignment (2-Phase)]")
# Phase 1: Run Method 2 logic (Strict Max Load, No Pref Restriction)
df_m3_ph1, load_m3_ph1 = execute_assignment_method_1_2(df_sections_to_assign, indicator_df, MAX_LOAD, PENALTY_RANK, method_type=2)
print("  -> Phase 1 (Strict Load) complete.")

# Phase 2: Run Method 3 logic (Flexible Load, Core Only)
df_m3, load_m3 = execute_assignment_method_3_4(df_m3_ph1, indicator_df, load_m3_ph1, MAX_LOAD, FLEXIBLE_MAX_LOAD_CAP, method_type=3)
_, load_summary_m3, unassigned_m3, kpi_m3 = calculate_kpis_and_summary(df_m3, indicator_df, MAX_LOAD, PENALTY_RANK, 'Method 3 (Flexible, Core Only)')
kpi_results['Method 3'] = kpi_m3

# --- Method 4: Flexible Max Load (CAP=FLEXIBLE_MAX_LOAD_CAP), All Sections Assignment ---
print("\n[Method 4: Flexible Max Load (Cap=5), All Sections Assignment (2-Phase)]")
# Phase 1: Run Method 2 logic (Strict Max Load, No Pref Restriction)
df_m4_ph1, load_m4_ph1 = execute_assignment_method_1_2(df_sections_to_assign, indicator_df, MAX_LOAD, PENALTY_RANK, method_type=2)
print("  -> Phase 1 (Strict Load) complete.")

# Phase 2: Run Method 4 logic (Flexible Load, All Sections)
df_m4, load_m4 = execute_assignment_method_3_4(df_m4_ph1, indicator_df, load_m4_ph1, MAX_LOAD, FLEXIBLE_MAX_LOAD_CAP, method_type=4)
_, load_summary_m4, unassigned_m4, kpi_m4 = calculate_kpis_and_summary(df_m4, indicator_df, MAX_LOAD, PENALTY_RANK, 'Method 4 (Flexible, All Sections)')
kpi_results['Method 4'] = kpi_m4

print("\n🎉 All four assignment methods completed.")

# --- Cell 18: Display Method 1 Results (Optional) ---
# Display results for Method 1
# print("\n--- Method 1 (Strict, Pref > 0) Assignment Results ---")
# print(df_m1.head())
# print("\n--- Method 1 (Strict, Pref > 0) Faculty Load Distribution (Load Summary) ---")
# print(load_summary_m1.to_markdown(index=False))
# print(f"\nNumber of unassigned sections: **{len(unassigned_m1)}**")

# --- Cell 19: Display Method 2 Results (Optional) ---
# Display results for Method 2
# print("\n--- Method 2 (Strict, No Pref) Assignment Results ---")
# print(df_m2.head())
# print("\n--- Method 2 (Strict, No Pref) Faculty Load Distribution (Load Summary) ---")
# print(load_summary_m2.to_markdown(index=False))
# print(f"\nNumber of unassigned sections: **{len(unassigned_m2)}**")

# --- Cell 20: Display Method 3 Results (Optional) ---
# Display results for Method 3
# print("\n--- Method 3 (Flexible, Core Only) Assignment Results ---")
# print(df_m3.head())
# print("\n--- Method 3 (Flexible, Core Only) Faculty Load Distribution (Load Summary) ---")
# print(load_summary_m3.to_markdown(index=False))
# print(f"\nNumber of unassigned sections: **{len(unassigned_m3)}**")

# --- Cell 21: Display Method 4 Results (Optional) ---
# Display results for Method 4
# print("\n--- Method 4 (Flexible, All Sections) Assignment Results ---")
# print(df_m4.head())
# print("\n--- Method 4 (Flexible, All Sections) Faculty Load Distribution (Load Summary) ---")
# print(load_summary_m4.to_markdown(index=False))
# print(f"\nNumber of unassigned sections: **{len(unassigned_m4)}**")

# --- Cell 22: Helper Function: create_kpi_summary_df ---
def create_kpi_summary_df(kpi_results):
    """
    Converts the dictionary of KPI results into a structured DataFrame.
    """
    # Convert the dictionary of results to a DataFrame
    kpi_df = pd.DataFrame(kpi_results).T

    # Set the 'Method' column as the index (if it's not already)
    if 'Method' in kpi_df.columns:
        kpi_df = kpi_df.set_index('Method')

    # Convert the index (method names) into a column to be the first column
    kpi_df = kpi_df.reset_index().rename(columns={'index': 'Method'})

    # Transpose the DataFrame to have KPIs as rows and Methods as columns
    kpi_df = kpi_df.set_index('Method').T.reset_index().rename(columns={'index': 'KPI'})

    return kpi_df

# --- Cell 23: Final KPI Comparison and Display ---
# --- Final KPI Comparison ---
final_kpi_summary = create_kpi_summary_df(kpi_results)

print("\n🌟 **Final KPI Comparison Across All Four Methods** 🌟")

# Convert rows ↔ columns (transpose)
final_kpi_summary = final_kpi_summary.T

# Use the first transposed row as column headers
final_kpi_summary.columns = final_kpi_summary.iloc[0]

# Remove the header row that was used for column names
final_kpi_summary = final_kpi_summary[1:].rename_axis(None, axis=1)

# print(final_kpi_summary.to_markdown()) # Uncomment this to print the final summary

# --- Cell 24: Export Results to Excel ---
#Insert a section: export the 4 outputs and the final_kpi_summary to an Excel file located...

# Use a Pandas ExcelWriter to save multiple DataFrames to different sheets
output_file = os.path.join(OUTPUT_DIR, "Assignment_Results_Summary.xlsx")

# Ensure the output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
    # Save the full assignment results
    df_m1.to_excel(writer, sheet_name='M1_Assignment_Details', index=False)
    df_m2.to_excel(writer, sheet_name='M2_Assignment_Details', index=False)
    df_m3.to_excel(writer, sheet_name='M3_Assignment_Details', index=False)
    df_m4.to_excel(writer, sheet_name='M4_Assignment_Details', index=False)

    # Save the load summaries
    load_summary_m1.to_excel(writer, sheet_name='M1_Faculty_Load', index=False)
    load_summary_m2.to_excel(writer, sheet_name='M2_Faculty_Load', index=False)
    load_summary_m3.to_excel(writer, sheet_name='M3_Faculty_Load', index=False)
    load_summary_m4.to_excel(writer, sheet_name='M4_Faculty_Load', index=False)

    # Save the final KPI comparison
    final_kpi_summary.to_excel(writer, sheet_name='Final_KPI_Comparison')

print(f"✅ All assignment results and KPIs saved to: {output_file}")


# --- Cell 25: Setup Gemini API Client (Colab Setup Adjusted) ---
# The following section has been adjusted to handle API key loading from a standard environment variable.
# If running locally, ensure you set the GEMINI_API_KEY environment variable.
# import os
# import google.genai as genai
# from google.colab import userdata # Colab-specific line commented out

# Set up the Gemini client
try:
    # Try to retrieve API Key from Secret Manager
    # API_KEY = userdata.get('GEMINI_API_KEY') # Colab-specific line commented out
    # os.environ['GEMINI_API_KEY'] = API_KEY # Colab-specific line commented out
    API_KEY = os.environ.get('GEMINI_API_KEY')
    if not API_KEY:
        raise ValueError('GEMINI_API_KEY not found in environment variables. Please set it.')
    client = genai.Client(api_key=API_KEY)
    print("Gemini client initialized successfully!")
except Exception as e:
    print("❗ Warning: Failed to initialize Gemini client.")
    print("Please make sure you have set the GEMINI_API_KEY environment variable.")
    print(f"Error details: {e}")

# --- Cell 26: Generate AI Analysis Report ---
# --- Generate AI Analysis Report ---
system_instruction = (
    "You are an expert AI analysis engine designed to evaluate the results of a faculty course assignment algorithm. "
    "Your task is to review the provided KPI (Key Performance Indicator) table and the faculty load summaries "
    "for four different assignment methods. "
    "Your goal is to provide a comprehensive, critical, and well-structured report that compares the methods "
    "based on the data, highlights trade-offs between coverage, load balancing, and faculty satisfaction, "
    "and offers a final, data-driven recommendation."
)

# Convert DataFrames to Markdown tables for the prompt
final_kpi_summary_md = final_kpi_summary.to_markdown()
load_summary_m1_md = load_summary_m1.to_markdown(index=False)
load_summary_m2_md = load_summary_m2.to_markdown(index=False)
load_summary_m3_md = load_summary_m3.to_markdown(index=False)
load_summary_m4_md = load_summary_m4.to_markdown(index=False)

prompt_template = f"""
**[INPUT DATA]**

--- Final KPI Comparison Across All Four Methods ---
{final_kpi_summary_md}

--- Method 1 (Strict, Pref > 0) Faculty Load Distribution (Load Summary) ---
{load_summary_m1_md}

--- Method 2 (Strict, No Pref) Faculty Load Distribution (Load Summary) ---
{load_summary_m2_md}

--- Method 3 (Flexible, Core Only) Faculty Load Distribution (Load Summary) ---
{load_summary_m3_md}

--- Method 4 (Flexible, All Sections) Faculty Load Distribution (Load Summary) ---
{load_summary_m4_md}

**[ANALYSIS TASK]**
Analyze the provided data from the four methods (Method 1-4) and generate a detailed report.
The analysis should focus on:
1.  **Coverage Analysis**: Compare Total, Core, and Elective Assigned Rates.
2.  **Quality Analysis (Faculty Satisfaction)**: Compare Avg Pref Rank, Overall Satisfaction Score, and No Preference Rate. Higher preference rank (lower number) is better, lower satisfaction score (lower number) is better.
3.  **Load Balance Analysis**: Compare Max Load Achieved and Load Std Dev. Lower Std Dev is better.
4.  **Trade-Offs**: Discuss the trade-offs, particularly between Method 2 (high satisfaction, low coverage) and Method 4 (full coverage, potentially lower satisfaction/higher load).

Conclude the findings and provide a final recommendation for the most balanced and effective assignment strategy.

**[REPORT FORMAT]**
The report must be professional, use clear section headings (like 1. Coverage Analysis, 2. Quality Analysis, etc.), and be written entirely in English.
"""

# Store the report content
report = "==================================================\n\n--- Gemini AI Generated Report ---\n"

try:
    # Call the Gemini API (use the latest flash model for strong performance)
    response = client.models.generate_content(
        model='gemini-2.5-flash', # recommended model
        contents=prompt_template,
        config=genai.types.GenerateContentConfig(
            system_instruction=system_instruction,
        )
    )
    report += response.text
    print(report)

except NameError:
    report += "❌ Gemini API call failed: name 'client' is not defined\n"
    report += "Error: Could not generate AI report due to API failure.\n"
    print(report)
except Exception as e:
    report += f"❌ Gemini API call failed: {e}\n"
    report += "Error: Could not generate AI report due to API failure.\n"
    print(report)

print("==================================================")

# --- Cell 27: Save AI Report ---
report_output_path = os.path.join(OUTPUT_DIR, "AI_Analysis_Report.txt")

os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(report_output_path, "w", encoding="utf-8") as f:
    f.write(report)

print(f"✅ AI analysis report saved to: {report_output_path}")