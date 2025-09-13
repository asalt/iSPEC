# test_io.py
import os
import pytest
import sqlite3
import io 
import tempfile
from ispec.io import io_file
from ispec.db.crud import Person, TableCRUD, Project, ProjectPerson
from ispec.db import init
#USE test_crud.py and test_data.py as example
#CONNECT TO DB FIRST - real or fake? - real :S
#TRIGGER THE IMPORT FILE
#----PREPARE TEST FILES
#THREE PROJECT FILES
#ONE PERSON FILE
# GET DATA :O
#  -w- z  z  z >:)
#Note: your brain keeps forgetting the context of what you're reading. improve your comments.

prjFile = 'test_ccsv_PROJECT.csv'
perFile = 'test_ccsv_PERSON.csv'

def previous():
    #CLEARS FILE BEFORE EVERY TEST
    #maybe can be summarized into a record?
    conn = sqlite3.connect(dbfile)
    cursor = conn.cursor()
    cursor.execute("DELETE from project")
    cursor.execute("DELETE from person")
    conn.commit()
    cursor.close()

    conn.close()

@pytest.mark.skip(reason="requires external test data files")
def test_import_file():
    db_path = "test3.db"
    if os.path.exists(db_path):
        os.unlink(db_path)
    #import project file first to test
    #Create project object first.
    #init.initialize_db(dbfile)
    io_file.import_file(prjFile, 'project', db_path) #path file must be a str
    io_file.import_file(perFile, 'person', db_path)
    io_file.import_file('ispec_people.xlsx', 'person' , db_path)
    io_file.import_file('ProjectsExport20250623.xlsx', 'project' , db_path)


#def test_import_multi_file(conn):
#def test_connect_prj_person_file(conn): # likely let the crud handle this part for you.

@pytest.mark.skip(reason="requires external test data files")
def test_import_comment():
    db_path = "test3.db"
    #import project file first to test
    #Create project object first.
    #init.initialize_db(dbfile)
    io_file.import_file('test_ccsv_COMMENT.csv','comment', db_path) #path file must be a str




@pytest.mark.skip(reason="io_file import requires refactor")
def test_import_person_file_with_extra_columns():
    s = b"ppl_AddedBy,ppl_Name_First,ppl_Name_Last,ppl_Phone,ppl_FavoriteColor\n" +\
        b"a,first,last2,222-333-4444,blue"
    f = tempfile.NamedTemporaryFile(suffix='.csv')
    f.write(s)
    f.seek(0)
    db_path = "./sandbox/test.db"
    if os.path.exists(db_path):
        os.unlink(db_path)

    init.initialize_db(file_path=db_path)
  
    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        query_res = conn.execute("select id from person").fetchall()
    initial_length = len(list(filter(None, query_res)))
    
    io_file.import_file(f.name, "person", db_file_path=db_path)


    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        query_res = conn.execute("select id from person").fetchall()
    assert len(query_res) == initial_length + 1

    io_file.import_file(f.name, "person", db_file_path=db_path)

    with sqlite3.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys = ON")
        query_res = conn.execute("select id from person").fetchall()
    assert len(query_res) == initial_length + 1



#-------important questions to consider
# How is person a file?
# How does project file connect to person file?
# For a project to successfully connect to a person when uploading files, does that person need to exist?
# How does current existing database work? - how does it connect between person and project?

#--- Answers 
# Based on the file types being inserted, the files are meant to be formatted to contain the information present in the columns for each table. So a person file would contain all that information like an excel sheet.
# though the ID's connect the person to the project, I am not 100% sure how the person can handle multiple projects, but we can make a project have many people, by copying the foreign key ID, and giving it to the person's foreign key id in their file.
# NO, the crud will create existing file.
# I believe person is a just an attribute that is added to the project, but are not linked to the people in the db

#------- ideas
# Can a person be a link from a project file? - How would one add these links to a project file and utilize them?
#for now, a project file likely has to be uploaded for it to have a foreign key, which can be modified into existing person files, or input in new files. so then that person will be 'assigned' to thagt project.
### - these ideas sort of require the HTML based code, but that part cannot come now, must come later.


