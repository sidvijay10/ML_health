#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:48:08 2023

@author: sidvijay
"""


import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd

# Load the provided CSV file
file_path = 'DGE_analysis.csv'
gene_data = pd.read_csv(file_path)

genes = pd.read_csv("gene_signature_Meningioma.csv")["Gene"].values.tolist()
gene_data.set_index('Gene', inplace=True)
selected_gene_data = gene_data.loc[genes]
top20 = selected_gene_data[:20]

top_genes = selected_gene_data.nlargest(20, 'correlation')

plt.figure(figsize=(12, 6))
sns.barplot(data=top20, x='Gene', y='correlation')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Correlation with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Correlation')
plt.tight_layout()



top_genes = selected_gene_data.nsmallest(20, 'rank sum test')

plt.figure(figsize=(12, 6))
sns.barplot(data=top20, x='Gene', y='rank sum test')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Rank Sum Test with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Rank Sum Test')
plt.tight_layout()

correlation_plot_path = 'Rank_Sum_Test.png'
plt.savefig(correlation_plot_path)
plt.show()


#gene_data = gene_data.sort_values(by='rank sum test')
top_genes = selected_gene_data.nsmallest(20, 'mean expression fold change (R/NR exp)')
plt.figure(figsize=(12, 6))
sns.barplot(data=top20, x='Gene', y='mean expression fold change (R/NR exp)')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Log Fold Change with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Rank Sum Test')
plt.tight_layout()

correlation_plot_path = 'Rank_Sum_Test.png'
plt.savefig(correlation_plot_path)
plt.show()
















# Display the first few rows of the dataframe to understand its structure
gene_data.head()


# Correlation Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=gene_data, x='Gene', y='correlation')
plt.xticks(rotation=90)
plt.title('Gene Correlation with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Correlation')
plt.tight_layout()

# Save the plot
correlation_plot_path = 'correlation_plot.png'
plt.savefig(correlation_plot_path)
plt.show()


top_genes = gene_data.nlargest(20, 'correlation')

# Creating a simplified correlation plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_genes, x='Gene', y='correlation')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Correlation with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Correlation')
plt.tight_layout()


#gene_data = gene_data.sort_values(by='rank sum test')
top_genes = gene_data.nsmallest(20, 'rank sum test')

# Creating a simplified correlation plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_genes, x='Gene', y='rank sum test')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Rank Sum Test with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Rank Sum Test')
plt.tight_layout()

# Save the plot
correlation_plot_path = 'Rank_Sum_Test.png'
plt.savefig(correlation_plot_path)
plt.show()


#gene_data = gene_data.sort_values(by='rank sum test')
top_genes = gene_data.nsmallest(20, 'mean expression fold change (R/NR exp)')

# Creating a simplified correlation plot
plt.figure(figsize=(12, 6))
sns.barplot(data=top_genes, x='Gene', y='rank sum test')
plt.xticks(rotation=90)
plt.title('Top 20 Genes Rank Sum Test with Meningioma Recurrence')
plt.xlabel('Gene')
plt.ylabel('Rank Sum Test')
plt.tight_layout()

# Save the plot
correlation_plot_path = 'Rank_Sum_Test.png'
plt.savefig(correlation_plot_path)
plt.show()









