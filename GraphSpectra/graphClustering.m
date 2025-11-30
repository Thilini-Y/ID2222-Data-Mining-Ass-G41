clear; clc;

% load dataset
E = csvread('example1.dat');

% support both 2 column and 3 column dataset
col1 = E(:,1);
col2 = E(:,2);

% conver column to 1,2 indexing
if min(min(col1, col2)) == 0
    col1 = col1 + 1;
    col2 = col2 + 1;
end

%number of nodes
n = max(max(col1, col2));

% Build affinity/adjacency matrix
A = sparse(col1, col2, 1, n, n);
A = A + A';              % make symmetric
A = double(A > 0);       % remove duplicates
A = full(A);

fprintf("Graph loaded with %d nodes and %d edges.\n", n, nnz(A)/2);


% normalized laplacian
degrees = sum(A,2);
D_inv_sqrt = diag(1 ./ sqrt(degrees));
D_inv_sqrt(isinf(D_inv_sqrt)) = 0;  % remove infinity

L = eye(n) - (D_inv_sqrt * A * D_inv_sqrt);


% 3. eigen values & vectors
fprintf("Computing eigenvalues...\n");

[V, D_eig] = eig(L);
eigenvalues = diag(D_eig);

% sort eigen values from smallest to largest
[sorted_vals, idx] = sort(eigenvalues, 'ascend');
sorted_vecs = V(:, idx);

% plot eigenvalues
figure;
plot(sorted_vals, 'o-');
title('Eigenvalues of Normalized Laplacian (L_{sym})');
xlabel('Index'); ylabel('\lambda_i');
grid on;


% determine K by looking for eigengap
fprintf("Check the eigenvalue plot and find the eigengap.\n");


% clustering
K = 4;  % choose after viewing eigenvalue plot

% Choose first K eigenvectors
X = sorted_vecs(:, 1:K);

% normalize rows
row_norms = sqrt(sum(X.^2, 2));
row_norms(row_norms == 0) = 1;
Y = X ./ row_norms;

% run k-means
fprintf("Running K-means...\n");
cluster_idx = kmeans(Y, K, 'Replicates', 10);


% visualize clusters
[~, order] = sort(cluster_idx);

figure;
spy(A(order, order));
title(sprintf('Adjacency Matrix Sorted by %d Clusters', K));

% Initial Graph (Before Clustering)
figure;
G = graph(A);
plot(G, 'Layout', 'force');
title('Initial Graph (Before Clustering)');

% Graph Colored by Clusters
figure;
G = graph(A);
p = plot(G, 'Layout', 'force');
title(sprintf('Graph Colored by %d Clusters', K));

cluster_colors = lines(K);
for i = 1:K
    highlight(p, find(cluster_idx == i), 'NodeColor', cluster_colors(i,:));
end