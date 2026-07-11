from pathlib import Path
import copy, csv, random
import numpy as np, torch
from torch.utils.data import DataLoader,TensorDataset

def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

class TorchRegressor:
    def __init__(self,network,config): self.network=network; self.config=config; self.history=[]; self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    def fit(self,train_data,train_target,valid_data=None,valid_target=None):
        c=self.config; seed_everything(c.get("seed",42)); self.network.to(self.device)
        train_x=np.array(train_data,dtype=np.float32,copy=True); train_y=np.array(train_target,dtype=np.float32,copy=True)
        loader=DataLoader(TensorDataset(torch.from_numpy(train_x),torch.from_numpy(train_y)),batch_size=c.get("batch_size",256),shuffle=True)
        valid_x=torch.as_tensor(np.array(valid_data,dtype=np.float32,copy=True),device=self.device) if valid_data is not None else None; valid_y=torch.as_tensor(np.array(valid_target,dtype=np.float32,copy=True),device=self.device) if valid_target is not None else None
        opt=torch.optim.Adam(self.network.parameters(),lr=c.get("learning_rate",1e-3),weight_decay=c.get("weight_decay",0)); loss_fn=torch.nn.L1Loss(); best=float("inf"); state=None; stale=0
        for epoch in range(1,c.get("max_epochs",5)+1):
            self.network.train(); losses=[]
            for x,y in loader:
                x,y=x.to(self.device),y.to(self.device); opt.zero_grad(); loss=loss_fn(self.network(x),y); loss.backward()
                if c.get("gradient_clip"): torch.nn.utils.clip_grad_norm_(self.network.parameters(),c["gradient_clip"])
                opt.step(); losses.append(loss.item())
            self.network.eval()
            with torch.no_grad(): val=float(loss_fn(self.network(valid_x),valid_y)) if valid_x is not None else float(np.mean(losses))
            self.history.append({"epoch":epoch,"train_loss":float(np.mean(losses)),"valid_mae":val})
            if val<best: best=val; state=copy.deepcopy(self.network.state_dict()); stale=0
            else: stale+=1
            if stale>=c.get("patience",3): break
        self.network.load_state_dict(state); return self
    def predict(self,data):
        self.network.eval(); x=torch.as_tensor(data,dtype=torch.float32,device=self.device)
        out=[]
        with torch.no_grad():
            for batch in x.split(self.config.get("batch_size",256)): out.append(self.network(batch).cpu().numpy())
        return np.concatenate(out)
    def save(self,path):
        from .models.mlp import MLPNet
        kind="mlp" if isinstance(self.network,MLPNet) else "gru"
        input_size=self.network.net[0].in_features if kind=="mlp" else self.network.gru.input_size
        torch.save({"state_dict":self.network.state_dict(),"config":self.config,"kind":kind,"input_size":input_size},path)
    @classmethod
    def load(cls,path):
        from .models.mlp import MLPNet
        from .models.gru import GRUNet
        checkpoint=torch.load(path,map_location="cpu",weights_only=False); c=checkpoint["config"]
        net=(MLPNet(checkpoint["input_size"],c.get("hidden_dims",[128,64]),c.get("dropout",.1)) if checkpoint["kind"]=="mlp" else GRUNet(checkpoint["input_size"],c.get("hidden_size",64),c.get("num_layers",1),c.get("dropout",0),c.get("bidirectional",False)))
        net.load_state_dict(checkpoint["state_dict"]); return cls(net,c)

def save_history(history,path):
    if not history:return
    with Path(path).open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=history[0]); w.writeheader(); w.writerows(history)
