import streamlit as st
import xml.etree.ElementTree as ET
import pandas as pd
import regex as re
#import openai

# Set your OpenAI API key
openai.api_key = 'your_openai_api_key'

def convert_to_dax_expression(expression):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Convert the following expression to a DAX expression and provide only the DAX code without any explanation: {expression}"}
            ],
            max_tokens=100
        )
        dax_expression = response.choices[0].message['content'].strip()
    except Exception as e:
        dax_expression = f"Error: {str(e)}"
    return dax_expression

def process_dataframe(df):
    df['powerbi formula'] = None
    
    for index, row in df.iterrows():
        if row['Rollup Aggregate'].lower() == 'total':
            expression = row['Expression']
            # Uncomment the following line to enable DAX conversion
            #dax_expression = convert_to_dax_expression(expression)
            #df.at[index, 'powerbi formula'] = dax_expression
    
    return df

def parse_model_path(model_path):
    package_start_index = model_path.find("@name='") + len("@name='")
    package_end_index = model_path.find("'", package_start_index)
    package_name = model_path[package_start_index:package_end_index]
    
    model_start_index = model_path.find("@name='", package_end_index) + len("@name='")
    model_end_index = model_path.find("'", model_start_index)
    model_name = model_path[model_start_index:model_end_index]
    
    return package_name, model_name

def parse_cognos_report(xml_content):
    root = ET.fromstring(xml_content)
    namespace = {'c': 'http://developer.cognos.com/schemas/report/16.2/'}
    
    report_name = root.find('c:reportName', namespace).text
    pages = root.findall('.//c:reportPages/c:page', namespace)
    num_pages = len(pages)
    
    model_path_element = root.find('c:modelPath', namespace)
    model_path = model_path_element.text if model_path_element is not None else 'No model path found'
    
    package_name, model_name = parse_model_path(model_path)
    
    queries = root.findall('.//c:queries/c:query', namespace)
    datasource_details = []
    
    for query in queries:
        query_name = query.get('name')
        columns = []
        detail_filters = []
        data_items = query.findall('.//c:selection/c:dataItem', namespace)
        
        for data_item in data_items:
            column_name = data_item.get('name')
            expression = data_item.find('c:expression', namespace).text
            rollup_aggregate = data_item.get('rollupAggregate', 'none')
            aggregate = data_item.get('aggregate', 'none')
            item_details = {
                'name': column_name,
                'expression': expression,
                'rollupAggregate': rollup_aggregate,
                'aggregate': aggregate
            }
            columns.append(item_details)
        
        detail_filters_nodes = query.findall('.//c:detailFilters/c:detailFilter', namespace)
        for filter_node in detail_filters_nodes:
            filter_expression = filter_node.find('c:filterExpression', namespace).text
            detail_filters.append({'expression': filter_expression})
        
        datasource_details.append({
            'query_name': query_name,
            'columns': columns,
            'detail_filters': detail_filters
        })
    
    page_details = []
    
    for page in pages:
        page_name = page.get('name')
        page_content = []
        
        lists = page.findall('.//c:list', namespace)
        for lst in lists:
            list_name = lst.get('name')
            ref_query = lst.get('refQuery')
            columns = []
            data_items = lst.findall('.//c:listColumnBody/c:contents/c:textItem/c:dataSource/c:dataItemValue', namespace)
            for data_item in data_items:
                columns.append(data_item.get('refDataItem'))
            page_content.append({
                'list_name': list_name,
                'ref_query': ref_query,
                'columns': columns
            })
        
        page_details.append({
            'page_name': page_name,
            'content': page_content
        })
    
    return report_name, num_pages, package_name, model_name, datasource_details, page_details

def aggregate_report_data(final_columns_df):
    # Group data by Report Name
    grouped = final_columns_df.groupby('Report Name')

    aggregated_data = []

    for report_name, group in grouped:
        page_names = group['Report Page Name'].unique()
        formatted_pages = []
        all_sources = set()
        all_filters = set()

        for page_name in page_names:
            page_group = group[group['Report Page Name'] == page_name]
            columns = page_group['Column Name'].unique()
            formatted_page = f"{page_name} Page: {', '.join(columns)}"
            formatted_pages.append(formatted_page)
            all_sources.update(page_group['Source'].unique())
            all_filters.update(page_group['Filters'].unique())

        aggregated_row = {
            'Report Name': report_name,
            'Report Page Name': ', '.join(page_names),
            'Column Name': '\n'.join(formatted_pages),
            'Source': ', '.join(filter(None, all_sources)),
            'Filters': '\n'.join(filter(None, all_filters))
        }

        aggregated_data.append(aggregated_row)

    return pd.DataFrame(aggregated_data)

def calculate_overlap_percentage(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    overlap = set1.intersection(set2)
    return len(overlap) / max(len(set1), len(set2)) * 100 if set1 and set2 else 0

def calculate_correlation_index(row1, row2):
    db_overlap = calculate_overlap_percentage(row1['Source'].split(', '), row2['Source'].split(', '))
    filter_overlap = calculate_overlap_percentage(row1['Filters'].split('\n'), row2['Filters'].split('\n'))
    metric_overlap = calculate_overlap_percentage(row1['Column Name'].split('\n'), row2['Column Name'].split('\n'))
    correlation_index = 0.5 * db_overlap + 0.2 * filter_overlap + 0.3 * metric_overlap
    return db_overlap, filter_overlap, metric_overlap, correlation_index

st.title("Cognos Report XML Comparer", help="This accelerator extracts the metadata from Cognos reports such as datasources used in report, columns used in report pages & much more ")

uploaded_files = st.file_uploader("Upload Cognos Report(s) in txt format)", type="txt", accept_multiple_files=True)

if uploaded_files:
    tabs = st.tabs([f"Report {i+1}" for i in range(len(uploaded_files))])
    
    final_columns_df = pd.DataFrame()

    for tab, uploaded_file in zip(tabs, uploaded_files):
        with tab:
            xml_content = uploaded_file.read().decode("utf-8")
            
            report_name, num_pages, package_name, model_name, datasource_details, page_details = parse_cognos_report(xml_content)
            
            st.write(f"**Report Name:** {report_name}")
            st.write(f"**Number of Pages:** {num_pages}")
            st.write(f"**Package Name:** {package_name}")
            st.write(f"**Model Name:** {model_name}")
            
            for datasource in datasource_details:
                if datasource['columns']:
                    columns_df = pd.DataFrame(datasource['columns'])
                
                if datasource['detail_filters']:
                    filters_df = pd.DataFrame(datasource['detail_filters'])
            
            for page in page_details:
                for content in page['content']:
                    if content['columns']:
                        columns_df = pd.DataFrame(content['columns'], columns=['Column Name'])
    
            rows = []
            for datasource in datasource_details:
                query_name = datasource['query_name']
                filters = ', '.join([f['expression'] for f in datasource['detail_filters']])
                for column in datasource['columns']:
                    used_in_page = "No"
                    page_name = "N/A"
                    for page in page_details:
                        for content in page['content']:
                            if content['ref_query'] == query_name and column['name'] in content['columns']:
                                used_in_page = "Yes"
                                page_name = page['page_name']
                                break
                    rows.append({
                        'Report Name': report_name,
                        'Query Name': query_name,
                        'Report Page Name': page_name,
                        'Column Name': column['name'],
                        'Expression': column['expression'],
                        'Rollup Aggregate': column['rollupAggregate'],
                        'Aggregate': column['aggregate'],
                        'Used in Report Page': used_in_page,
                        'Filters': filters
                    })
            final_columns_df = pd.concat([final_columns_df, pd.DataFrame(rows)], ignore_index=True)
    
    pattern = r'\[([^\]]+)\]\.\[([^\]]+)\]\.\[([^\]]+)\]'

    def extract_source(text):
        match = re.search(pattern, text)
        if match:
            return f"{match.group(1)}.{match.group(2)}"
        else:
            return ''

    final_columns_df['Source'] = final_columns_df['Expression'].apply(extract_source)

    final_columns_df = final_columns_df[[
        'Report Name', 'Report Page Name', 'Query Name', 'Column Name', 
        'Expression', 'Rollup Aggregate', 'Aggregate', 'Used in Report Page', 'Filters', 'Source'
    ]]

    final_columns_df = process_dataframe(final_columns_df)

    aggregated_df = aggregate_report_data(final_columns_df)

    # Calculate overlaps and correlation index
    comparison_rows = []
    report_pairs = [(i, j) for i in range(len(aggregated_df)) for j in range(i+1, len(aggregated_df))]

    for i, j in report_pairs:
        row1 = aggregated_df.iloc[i]
        row2 = aggregated_df.iloc[j]
        db_overlap, filter_overlap, metric_overlap, correlation_index = calculate_correlation_index(row1, row2)
        comparison_rows.append({
            'Report_ID_1': row1['Report Name'],
            'Report_Name_1': row1['Report Name'],
            'Report_ID_2': row2['Report Name'],
            'Report_Name_2': row2['Report Name'],
            'Database_Attribute_Overlap_Percentage': db_overlap,
            'Filters_Overlap_Percentage': filter_overlap,
            'Metrics_Overlap_Percentage': metric_overlap,
            'Report_Correlation_Index_Score': correlation_index
        })

    comparison_df = pd.DataFrame(comparison_rows)

    # Calculate overall score
    overall_db_overlap = comparison_df['Database_Attribute_Overlap_Percentage'].mean()
    overall_filter_overlap = comparison_df['Filters_Overlap_Percentage'].mean()
    overall_metric_overlap = comparison_df['Metrics_Overlap_Percentage'].mean()
    overall_score = comparison_df['Report_Correlation_Index_Score'].mean()

    st.info("Report Analysis")
    st.dataframe(aggregated_df)
    st.dataframe(comparison_df)
    #st.write(f"**Overall Correlation Index Score:** {overall_score}")
    #st.code('0.5 * db_overlap + 0.2 * filter_overlap + 0.3 * metric_overlap')
    st.write(f"- **Database Attribute Overlap Percentage:** {overall_db_overlap}")
    st.write(f"- **Filters Overlap Percentage:** {overall_filter_overlap}")
    st.write(f"- **Metrics Overlap Percentage:** {overall_metric_overlap}")

    csv = aggregated_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Aggregated Data as CSV",
        data=csv,
        file_name='aggregated_report_data.csv',
        mime='text/csv',
    )

    csv_comparison = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="Download Comparison Data as CSV",
        data=csv_comparison,
        file_name='comparison_report_data.csv',
        mime='text/csv',
    )
else:
    st.write("Please upload one or more Cognos reports in txt format.")
