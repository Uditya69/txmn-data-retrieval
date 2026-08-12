# High Level Search Logic

## Steps Involved
1. Users Input --> Tokens (Using Taxmann Query Analyzer and Processor Query Functions)
2. Construction of ES Query

## Construction of ES Query

1. We use multiple fields like: heading, subheading, headnotes, searchboosttext, fullcontent

2. Each token should be present in atleast one of the field. And all tokens should satisfy this condition
  
    ```
        Example: Tokens: [Section 6, Income, Supreme Court]

        MUST:[
            should:{section 6 : One of the fields},
            should: {Income: One of the fields}
            should: {Supreme Court: One of the fields}
        ]
    ```

## Users Query Processing

1. Convert the User Query into tokens using Query Analyzer Function:  
  [Detailed Logic is in file name: services/queryAnalyzer.js  --> taxmannQueryAnalyzer Function]  
  Ip: Raw User Query  
  Op: Tokens

  Query Analyzer Function takes care of following things.  
  a. Removes special characters other than %  
  b. Handle Citation Formats  
  c. Remove Multiple Stop words (Ex. In case of)  
  d. Grouping double quoted phrases together as a single token  

  Ex. Section 6 of "Income India" in case of Supreme court.  
  Op: ["Section", "6", "of", "Income India", "Supreme", "court"]  

2. Processor Query Function:  
  Ip: Query Tokens from Query Analyzer Function  
  Op: Grouped, Merged, using token.js file

3. We check whether token present in the token.js file.  
  ```
  Token Structure in token.js file:  

  SNO: <Sequence in the file>,
  KeyWordExact: <key name in the json file>,
  TypeCode: <type of token code>,
  ZoneType: <type of token value>,
  SearchText: <text to be used during searching if the key matched>,
  Params: < tagnumber : slop value : boost : group id to which boosting to be applied>

  ```

4. We have two flows:   
a. Token present in token.js file  
b. Token not present in token.js file

5. Token Present: Logic based on the token Type  
```
a. KEYWORD / KeyWordOnly / KeyWordType2
  Goal: Match with a following number (e.g., Section + 5).

  Result:
  If a number is found → merge them.
  If not:
    KEYWORD → keep as standalone keyword.
    KeyWordOnly → ignore.

b. Zone / Country
  Goal: Treat as single element.
  
  Result:
  Add directly to the final token list.
  Zone → set as the primary tag.
  Country → do not set as the primary tag.

c. High Court
  Goal: Look ahead for a specific court name (e.g., Delhi, Bombay).

  Result:
  If found → merge them (e.g., HIGH COURT DELHI).
  If not → backtrack.

d. KeyWordOrStopWord
  Goal: Based on type. Classify

  Result:
  Followed by a number → merge and keep.
  Otherwise → treat as a stop word and ignore.

e. StopWord / Synonym
  Goal: Skip these tokens.

  Result:
  Do not add anything to the final token list.

f. Journal
  Goal: Ensure a Journal name is never discarded.
  
  Result:
  If valid citation details are present → combine them.
  Otherwise → keep the Journal as a standalone token.

g. Court
  Goal: Combine a court location with its court type (e.g., Delhi + High Court).

  Result:
  If combined → create a targeted token such as DELHI HIGH COURT.
  If not combined → backtrack and keep the location as a standalone search term.
```

6. Token not present  
```
We check whether it valid date format, date month, month year, circular format, jounrnal format etc. and group the tokens accordinly
```

After dividing into tokens , we need to construct the ES low level query