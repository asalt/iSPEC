
CREATE TABLE IF NOT EXISTS person (
  ppl_id INTEGER PRIMARY KEY,
  ppl_AddedBy TEXT,
  ppl_CreationTS TEXT,
  ppl_ModificationTS TEXT,
  -- ppl_Name TEXT
  ppl_Name_First TEXT,
  ppl_Name_Last TEXT,
  ppl_Domain TEXT,
  ppl_Email TEXT,
  ppl_Phone TEXT,
  ppl_PI TEXT,
  ppl_Institution TEXT,
  ppl_Center TEXT,
  ppl_Department TEXT,
  ppl_Status TEXT,
  ppl_Roles_PI TEXT,
  ppl_Roles_CoreUser TEXT,
  ppl_Roles_CoreStaff TEXT,
  ppl_Roles_Collaborator TEXT,
  ppl_Roles TEXT
);

CREATE TABLE IF NOT EXISTS project (
  prj_id INTEGER PRIMARY KEY,
  prj_AddedBy TEXT,
  prj_CreationTS TEXT,
  prj_ModificationTS TEXT,
  prj_PRJ_DisplayID TEXT,
  prj_PRJ_DisplayTitle TEXT,
  prj_iLabs_RequestName TEXT,
  prj_ProjectTitle TEXT,
  prj_RnD TEXT,
  prj_CancerRelevance TEXT,
  prj_CPRIT_RFP TEXT,
  prj_PI TEXT,
  prj_Project_LabContact TEXT,
  prj_Project_LabPersonnel TEXT,
  prj_MSPC_Leader TEXT,
  prj_MSPC_Personnel_Primary TEXT,
  prj_MSPC_Personnel TEXT,
  prj_MSPC_Personnel_Analysis TEXT,
  prj_Services_Type TEXT,
  prj_Services TEXT,
  prj_GrantSupport TEXT,
  prj_ProjectDomain TEXT,
  prj_ProjectBackground TEXT,
  prj_ProjectSuggestions2Customer TEXT,
  prj_ProjectSamples TEXT,
  prj_ProjectCoreTasks TEXT,
  prj_ProjectQuestions TEXT,
  prj_ProjectDescription TEXT,
  prj_Project_SampleType TEXT,
  prj_Project_FuturePossibilities TEXT,
  prj_RequireGelPix TEXT,
  prj_Experiments_rf TEXT,
  prj_RelatedProjects_rf TEXT,
  prj_Comments_Meeting TEXT,
  -- prj_FLAG_CommentType_Meeting  TEXT
  prj_Comments_Billing TEXT,
  -- prj_FLAG_CommentType_Billing TEXT
  prj_Comments_iLAB TEXT,
  -- prj_FLAG_CommentType_iLAB TEXT
  prj_Status TEXT,
  prj_Date_Submitted TEXT,
  prj_Date_LysatePreparation TEXT,
  prj_Date_MSPreparation TEXT,
  prj_RequireGelPix_Sent TEXT,
  prj_RequireGelPix_Confirmed TEXT,
  prj_Date_Analysis TEXT,
  prj_Comments_Specific TEXT,
  prj_Comments_General TEXT,
  prj_Date_Review TEXT,
  prj_Date_Closed TEXT,
  prj_Comments_Review TEXT,
  prj_IncludeHandouts TEXT,
  prj_ProjectCostExplanation TEXT,
  -- prj_ProjectPriceLevel TEXT
  -- prj_Service_365_Experiments TEXT
  prj_Service_365_Analysis TEXT,
  prj_Service_APMS_Experiments TEXT,
  prj_Service_APMS_Analysis TEXT,
  prj_Service_PerBand_Experiments TEXT,
  prj_Service_PerBand_Analysis TEXT,
  prj_Service_PTM_Analysis TEXT,
  prj_Service_DesignHours TEXT,
  prj_ProjectPrice TEXT,
  prj_Quote TEXT,
  prj_Invoice TEXT,
  prj_Invoice_Date TEXT,
  prj_Billing_Date TEXT,
  prj_PaymentReceived TEXT,
  prj_ProjectDomainMSPC TEXT,
  prj_Project_SampleHandling TEXT,
  prj_ExpCount INTEGER,
  prj_ExpRunCount INTEGER,
  prj_MSFilesCount INTEGER,
  prj_MSRunTime REAL,
  prj_MSPCEmail TEXT,
  -- prj_FileAttachments TEXT
  -- dev_QuoteName TEXT
  -- prj_StatusUsed TEXT
  prj_PaidPrice REAL,
  prj_ProjectCostMinimum REAL,
  prj_ProjectCostMaximum REAL,
  prj_Current_FLAG INTEGER,
  prj_Billing_ReadyToBill INTEGER
);

CREATE TABLE IF NOT EXISTS letter_of_support (
  id INTEGER PRIMARY KEY
);

CREATE TABLE IF NOT EXISTS project_person (
  id INTEGER PRIMARY KEY NOT NULL,
  project_id INTEGER NOT NULL,
  person_id INTEGER NOT NULL,
  FOREIGN KEY(project_id) REFERENCES project(prj_id),
  FOREIGN KEY(person_id) REFERENCES project(ppl_id)
);
