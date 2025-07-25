import dgl
import torch
import numpy as np
import torch.nn as nn
import dgl.function as fn
import torch.nn.functional as F

class DHGNNLayerDGL(nn.Module):
    def __init__(self, device, in_dim, out_dim, num_heads, dropout):
        super(DHGNNLayerDGL, self).__init__()
        self.device = device
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        
        self.node_to_edge_layers = nn.ModuleDict({
            'drug-gene': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
            'disease-gene': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
            'drug-phenotype': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
            'disease-phenotype': NodeToHyperedgeLayer(in_dim, out_dim, num_heads),
        })
        
        self.edge_to_node_layers = nn.ModuleDict({
            'gene-drug': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
            'phenotype-drug': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
            'gene-disease': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
            'phenotype-disease': HyperedgeToNodeLayer(out_dim, out_dim, num_heads),
        })
        
        self.gate_drug = nn.Linear(out_dim * 2, out_dim)
        self.gate_disease = nn.Linear(out_dim * 2, out_dim)
        self.gate_gene = nn.Linear(out_dim * 2, out_dim)
        self.gate_phenotype = nn.Linear(out_dim * 2, out_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(out_dim)
        self.layer_norm2 = nn.LayerNorm(out_dim)

        self._init_parameters()
    
    def _init_parameters(self):
        gate_layers = [self.gate_drug, self.gate_disease, self.gate_gene, self.gate_phenotype]
        for gate_layer in gate_layers:
            nn.init.xavier_uniform_(gate_layer.weight)
            if gate_layer.bias is not None:
                nn.init.zeros_(gate_layer.bias)

    def forward(self, g, node_feats):
        edge_feats = {}
        
        if 'gene' in node_feats:
            num_genes = node_feats['gene'].shape[0]
            gene_messages_from_drug = torch.zeros(num_genes, self.out_dim).to(self.device)
            gene_messages_from_disease = torch.zeros(num_genes, self.out_dim).to(self.device)
            
            has_drug_connection = torch.zeros(num_genes, dtype=torch.bool).to(self.device)
            has_disease_connection = torch.zeros(num_genes, dtype=torch.bool).to(self.device)
            
            if 'drug-gene' in g.etypes and g.num_edges('drug-gene') > 0:
                subg = g['drug-gene']
                _, gene_indices = subg.edges()
                unique_gene_indices = torch.unique(gene_indices)
                has_drug_connection[unique_gene_indices] = True
                
                gene_msg = self.node_to_edge_layers['drug-gene'](
                    subg, (node_feats['drug'], node_feats['gene']))
                
                if gene_msg is not None:
                    gene_messages_from_drug = gene_msg
            
            if 'disease-gene' in g.etypes and g.num_edges('disease-gene') > 0:
                subg = g['disease-gene']
                _, gene_indices = subg.edges()
                unique_gene_indices = torch.unique(gene_indices)
                has_disease_connection[unique_gene_indices] = True
                
                gene_msg = self.node_to_edge_layers['disease-gene'](
                    subg, (node_feats['disease'], node_feats['gene']))
                
                if gene_msg is not None:
                    gene_messages_from_disease = gene_msg
            
            gene_final_messages = torch.zeros(num_genes, self.out_dim).to(self.device)
            
            both_connected = has_drug_connection & has_disease_connection 
            if both_connected.any():
                gate_input = torch.cat([
                    gene_messages_from_drug[both_connected],
                    gene_messages_from_disease[both_connected]
                ], dim=1)
                g_drug = torch.sigmoid(self.gate_drug(gate_input))
                g_disease = torch.sigmoid(self.gate_disease(gate_input))
                
                g_sum = g_drug + g_disease + 1e-8
                g_drug = g_drug / g_sum
                g_disease = g_disease / g_sum
                
                gene_final_messages[both_connected] = (
                    g_drug * gene_messages_from_drug[both_connected] +
                    g_disease * gene_messages_from_disease[both_connected])
            
            only_drug = has_drug_connection & ~has_disease_connection
            if only_drug.any():
                gene_final_messages[only_drug] = gene_messages_from_drug[only_drug]
            
            only_disease = ~has_drug_connection & has_disease_connection
            if only_disease.any():
                gene_final_messages[only_disease] = gene_messages_from_disease[only_disease]
            
            edge_feats['gene'] = gene_final_messages

        if 'phenotype' in node_feats:
            num_phenotypes = node_feats['phenotype'].shape[0]
            pheno_messages_from_drug = torch.zeros(num_phenotypes, self.out_dim).to(self.device)
            pheno_messages_from_disease = torch.zeros(num_phenotypes, self.out_dim).to(self.device)
            
            has_drug_connection = torch.zeros(num_phenotypes, dtype=torch.bool).to(self.device)
            has_disease_connection = torch.zeros(num_phenotypes, dtype=torch.bool).to(self.device)
            
            if 'drug-phenotype' in g.etypes and g.num_edges('drug-phenotype') > 0:
                subg = g['drug-phenotype']
                _, pheno_indices = subg.edges()
                unique_pheno_indices = torch.unique(pheno_indices)
                has_drug_connection[unique_pheno_indices] = True
                
                pheno_msg = self.node_to_edge_layers['drug-phenotype'](
                    subg, (node_feats['drug'], node_feats['phenotype']))
                
                if pheno_msg is not None:
                    pheno_messages_from_drug = pheno_msg
            
            if 'disease-phenotype' in g.etypes and g.num_edges('disease-phenotype') > 0:
                subg = g['disease-phenotype']
                _, pheno_indices = subg.edges()
                unique_pheno_indices = torch.unique(pheno_indices)
                has_disease_connection[unique_pheno_indices] = True
                
                pheno_msg = self.node_to_edge_layers['disease-phenotype'](
                    subg, (node_feats['disease'], node_feats['phenotype']))
                if pheno_msg is not None:
                    pheno_messages_from_disease = pheno_msg
            
            pheno_final_messages = torch.zeros(num_phenotypes, self.out_dim).to(self.device)
            
            both_connected = has_drug_connection & has_disease_connection

            if both_connected.any():
                gate_input = torch.cat([
                    pheno_messages_from_drug[both_connected],
                    pheno_messages_from_disease[both_connected]
                ], dim=1)
                g_drug = torch.sigmoid(self.gate_drug(gate_input))
                g_disease = torch.sigmoid(self.gate_disease(gate_input))
                
                g_sum = g_drug + g_disease + 1e-8
                g_drug = g_drug / g_sum
                g_disease = g_disease / g_sum
                
                pheno_final_messages[both_connected] = (
                    g_drug * pheno_messages_from_drug[both_connected] +
                    g_disease * pheno_messages_from_disease[both_connected])
            
            only_drug = has_drug_connection & ~has_disease_connection
            if only_drug.any():
                pheno_final_messages[only_drug] = pheno_messages_from_drug[only_drug]
            
            only_disease = ~has_drug_connection & has_disease_connection
            if only_disease.any():
                pheno_final_messages[only_disease] = pheno_messages_from_disease[only_disease]
            
            edge_feats['phenotype'] = pheno_final_messages

        if 'gene' in edge_feats and 'gene' in node_feats:
            edge_feats['gene'] = self.layer_norm1(
                node_feats['gene'] + self.dropout(F.gelu(edge_feats['gene'])))
            
        if 'phenotype' in edge_feats and 'phenotype' in node_feats:
            edge_feats['phenotype'] = self.layer_norm1(
                node_feats['phenotype'] + self.dropout(F.gelu(edge_feats['phenotype'])))
        
        edge_feats['drug'] = node_feats['drug']
        edge_feats['disease'] = node_feats['disease']
        
        new_node_feats = {}
        
        if 'drug' in node_feats:
            num_drugs = node_feats['drug'].shape[0]
            drug_messages_from_gene = torch.zeros(num_drugs, self.out_dim).to(self.device)
            drug_messages_from_pheno = torch.zeros(num_drugs, self.out_dim).to(self.device)
            
            has_gene_connection = torch.zeros(num_drugs, dtype=torch.bool).to(self.device)
            has_pheno_connection = torch.zeros(num_drugs, dtype=torch.bool).to(self.device)
            
            if 'gene-drug' in g.etypes and g.num_edges('gene-drug') > 0:
                subg = g['gene-drug']
                _, drug_indices = subg.edges()
                unique_drug_indices = torch.unique(drug_indices)
                has_gene_connection[unique_drug_indices] = True
                
                drug_msg = self.edge_to_node_layers['gene-drug'](
                    subg, (edge_feats['gene'], edge_feats['drug']))
                
                if drug_msg is not None:
                    drug_messages_from_gene = drug_msg
            
            if 'phenotype-drug' in g.etypes and g.num_edges('phenotype-drug') > 0:
                subg = g['phenotype-drug']
                _, drug_indices = subg.edges()
                unique_drug_indices = torch.unique(drug_indices)
                has_pheno_connection[unique_drug_indices] = True
                
                drug_msg = self.edge_to_node_layers['phenotype-drug'](
                    subg, (edge_feats['phenotype'], edge_feats['drug']))
                
                if drug_msg is not None:
                    drug_messages_from_pheno = drug_msg
            
            drug_final_messages = torch.zeros(num_drugs, self.out_dim).to(self.device)
            
            both_connected = has_gene_connection & has_pheno_connection
            
            if both_connected.any():
                gate_input = torch.cat([
                    drug_messages_from_gene[both_connected],
                    drug_messages_from_pheno[both_connected]
                ], dim=1)
                g_gene = torch.sigmoid(self.gate_gene(gate_input))
                g_phenotype = torch.sigmoid(self.gate_phenotype(gate_input))
                
                g_sum = g_gene + g_phenotype + 1e-8
                g_gene = g_gene / g_sum
                g_phenotype = g_phenotype / g_sum
                
                drug_final_messages[both_connected] = (
                    g_gene * drug_messages_from_gene[both_connected] +
                    g_phenotype * drug_messages_from_pheno[both_connected])
            
            only_gene = has_gene_connection & ~has_pheno_connection
            if only_gene.any():
                drug_final_messages[only_gene] = drug_messages_from_gene[only_gene]
            
            only_pheno = ~has_gene_connection & has_pheno_connection
            if only_pheno.any():
                drug_final_messages[only_pheno] = drug_messages_from_pheno[only_pheno]
            
            new_node_feats['drug'] = drug_final_messages

        if 'disease' in node_feats:
            num_diseases = node_feats['disease'].shape[0]
            disease_messages_from_gene = torch.zeros(num_diseases, self.out_dim).to(self.device)
            disease_messages_from_pheno = torch.zeros(num_diseases, self.out_dim).to(self.device)
            
            has_gene_connection = torch.zeros(num_diseases, dtype=torch.bool).to(self.device)
            has_pheno_connection = torch.zeros(num_diseases, dtype=torch.bool).to(self.device)
            
            if 'gene-disease' in g.etypes and g.num_edges('gene-disease') > 0:
                subg = g['gene-disease']
                _, disease_indices = subg.edges()
                unique_disease_indices = torch.unique(disease_indices)
                has_gene_connection[unique_disease_indices] = True
                
                disease_msg = self.edge_to_node_layers['gene-disease'](
                    subg, (edge_feats['gene'], edge_feats['disease']))
                
                if disease_msg is not None:
                    disease_messages_from_gene = disease_msg
            
            if 'phenotype-disease' in g.etypes and g.num_edges('phenotype-disease') > 0:
                subg = g['phenotype-disease']
                _, disease_indices = subg.edges()
                unique_disease_indices = torch.unique(disease_indices)
                has_pheno_connection[unique_disease_indices] = True
                
                disease_msg = self.edge_to_node_layers['phenotype-disease'](
                    subg, (edge_feats['phenotype'], edge_feats['disease']))
                
                if disease_msg is not None:
                    disease_messages_from_pheno = disease_msg
            
            disease_final_messages = torch.zeros(num_diseases, self.out_dim).to(self.device)
            
            both_connected = has_gene_connection & has_pheno_connection
            if both_connected.any():
                gate_input = torch.cat([
                    disease_messages_from_gene[both_connected],
                    disease_messages_from_pheno[both_connected]
                ], dim=1)
                g_gene = torch.sigmoid(self.gate_gene(gate_input))
                g_phenotype = torch.sigmoid(self.gate_phenotype(gate_input))
                
                g_sum = g_gene + g_phenotype + 1e-8
                g_gene = g_gene / g_sum
                g_phenotype = g_phenotype / g_sum
                
                disease_final_messages[both_connected] = (
                    g_gene * disease_messages_from_gene[both_connected] +
                    g_phenotype * disease_messages_from_pheno[both_connected])
            
            only_gene = has_gene_connection & ~has_pheno_connection
            
            if only_gene.any():
                disease_final_messages[only_gene] = disease_messages_from_gene[only_gene]
            
            only_pheno = ~has_gene_connection & has_pheno_connection
            if only_pheno.any():
                disease_final_messages[only_pheno] = disease_messages_from_pheno[only_pheno]
            
            new_node_feats['disease'] = disease_final_messages

        if 'drug' in new_node_feats and 'drug' in node_feats:
            new_node_feats['drug'] = self.layer_norm2(
                node_feats['drug'] + self.dropout(F.gelu(new_node_feats['drug'])))

        if 'disease' in new_node_feats and 'disease' in node_feats:
            new_node_feats['disease'] = self.layer_norm2(
                node_feats['disease'] + self.dropout(F.gelu(new_node_feats['disease'])))
        
        new_node_feats['gene'] = edge_feats.get('gene', node_feats.get('gene'))
        new_node_feats['phenotype'] = edge_feats.get('phenotype', node_feats.get('phenotype'))
        
        return new_node_feats

class NodeToHyperedgeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(NodeToHyperedgeLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)
        self.W_V = nn.Linear(in_dim, out_dim)

        self._init_parameters()
    
    def _init_parameters(self):
        for linear in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
    def forward(self, g, feat):
        with g.local_scope():
            if isinstance(feat, tuple):
                h_src, h_dst = feat
            else:
                h_src = feat
                h_dst = feat

            Q = self.W_Q(h_dst).view(-1, self.num_heads, self.out_dim // self.num_heads)
            K = self.W_K(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            V = self.W_V(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            
            g.dstdata['Q'] = Q
            g.srcdata['K'] = K
            g.srcdata['V'] = V
            
            g.apply_edges(fn.u_dot_v('K', 'Q', 'score'))
            g.edata['score'] = g.edata['score'] / np.sqrt(self.out_dim // self.num_heads)
            
            g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['score'])
            
            g.update_all(
                fn.u_mul_e('V', 'a', 'm'),
                fn.sum('m', 'h'))
            
            h_out = g.dstdata['h'].view(-1, self.out_dim)
            
            return h_out

class HyperedgeToNodeLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads):
        super(HyperedgeToNodeLayer, self).__init__()
        
        self.num_heads = num_heads
        self.out_dim = out_dim
        
        self.W_Q = nn.Linear(in_dim, out_dim)
        self.W_K = nn.Linear(in_dim, out_dim)
        self.W_V = nn.Linear(in_dim, out_dim)
        self._init_parameters()
    
    def _init_parameters(self):
        for linear in [self.W_Q, self.W_K, self.W_V]:
            nn.init.xavier_uniform_(linear.weight)
            if linear.bias is not None:
                nn.init.zeros_(linear.bias)
        
    def forward(self, g, feat):
        with g.local_scope():
            if isinstance(feat, tuple):
                h_src, h_dst = feat
            else:
                h_src = feat
                h_dst = feat

            Q = self.W_Q(h_dst).view(-1, self.num_heads, self.out_dim // self.num_heads)
            K = self.W_K(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            V = self.W_V(h_src).view(-1, self.num_heads, self.out_dim // self.num_heads)
            
            g.dstdata['Q'] = Q
            g.srcdata['K'] = K
            g.srcdata['V'] = V
            
            g.apply_edges(fn.u_dot_v('K', 'Q', 'score'))
            g.edata['score'] = g.edata['score'] / np.sqrt(self.out_dim // self.num_heads)
            g.edata['a'] = dgl.ops.edge_softmax(g, g.edata['score'])
            
            g.update_all(
                fn.u_mul_e('V', 'a', 'm'),
                fn.sum('m', 'h'))
            
            h_out = g.dstdata['h'].view(-1, self.out_dim)
            
            return h_out

class DecoupledHGNNDGL(nn.Module):
    def __init__(self, device, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout):
        super(DecoupledHGNNDGL, self).__init__()

        self.input_projs = nn.ModuleDict({
            'drug': nn.Linear(in_dim, hidden_dim),
            'disease': nn.Linear(in_dim, hidden_dim),
            'gene': nn.Linear(in_dim, hidden_dim),
            'phenotype': nn.Linear(in_dim, hidden_dim),
        })
        
        self.layers = nn.ModuleList([
            DHGNNLayerDGL(device, hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.output_projs = nn.ModuleDict({
            'drug': nn.Linear(hidden_dim, out_dim),
            'disease': nn.Linear(hidden_dim, out_dim),
        })

        self._init_parameters()
    
    def _init_parameters(self):
        for proj_dict in [self.input_projs, self.output_projs]:
            for proj in proj_dict.values():
                nn.init.xavier_uniform_(proj.weight)
                nn.init.zeros_(proj.bias)
        
    def forward(self, g):
        h = {
            ntype: self.input_projs[ntype](g.nodes[ntype].data['feat'])
            for ntype in g.ntypes}
        
        for layer in self.layers:
            h = layer(g, h)
            
        out_feats = {
            'drug': self.output_projs['drug'](h['drug']),
            'disease': self.output_projs['disease'](h['disease'])}
        
        return out_feats

class DrugDiseasePredictor(nn.Module):
    def __init__(self, in_dim, hidden_dim, dropout):
        super(DrugDiseasePredictor, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim), 
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.Linear(hidden_dim // 2, 1))
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def _multiple_operator(self, a, b):
        return a * b

    def _rotate_operator(self, a, b):
        a_re, a_im = a.chunk(2, dim=-1)
        b_re, b_im = b.chunk(2, dim=-1)
        message_re = a_re * b_re - a_im * b_im
        message_im = a_re * b_im + a_im * b_re
        message = torch.cat([message_re, message_im], dim=-1)
        return message
                
    def forward(self, drug_embeds, disease_embeds):
        m_result = self._multiple_operator(drug_embeds, disease_embeds)
        r_result = self._rotate_operator(drug_embeds, disease_embeds)
        combined = torch.cat([drug_embeds, disease_embeds, m_result, r_result], dim=1)

        scores = self.mlp(combined).squeeze(-1)
        
        return torch.sigmoid(scores)

class DrugRepositioningModel(nn.Module):
    def __init__(self, hypergraph, device, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout):
        super(DrugRepositioningModel, self).__init__()
        
        self.g = hypergraph
        
        self.dhgnn = DecoupledHGNNDGL(device, in_dim, hidden_dim, out_dim, num_heads, num_layers, dropout)
        
        self.predictor = DrugDiseasePredictor(in_dim, hidden_dim, dropout=dropout)
    
    def forward(self, drug_indices, disease_indices):
        node_embeddings = self.dhgnn(self.g)
        
        drug_embeds = node_embeddings['drug'][drug_indices]

        disease_embeds = node_embeddings['disease'][disease_indices]
        
        scores = self.predictor(drug_embeds, disease_embeds)
        
        return scores

    def save(self, checkpoint_dir):
        torch.save(self.state_dict(), checkpoint_dir)
        
    def load(self, checkpoint_dir):
        self.load_state_dict(torch.load(checkpoint_dir))