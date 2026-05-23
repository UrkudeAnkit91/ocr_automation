import { Component, OnInit, OnDestroy, signal, inject } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { ApiService, LogEntry } from './services/api.service';

interface Record {
  id: number;
  created_at: string;
  updated_at: string;
  first_name: string; last_name: string; email: string; ssn: string;
  phone: string; bank_name: string; account_no: string; loan_amount: string;
  address: string; city: string; state: string; zip: string;
  dob: string; licence_no: string; licence_state: string; ip: string;
  raw_ocr: string;
}

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  template: `
<div class="app">
  <header>
    <h1>📋 OCR Records</h1>
    <div class="header-info">
      <span class="badge" [class.busy]="status()?.busy">{{ status()?.busy ? 'Busy' : 'Idle' }}</span>
      <span class="count">{{ totalRecords }} records</span>
    </div>
  </header>

  <div class="toolbar">
    <button class="btn btn-primary" (click)="triggerOcr()" [disabled]="status()?.busy">📷 New OCR</button>
    <input class="search" [(ngModel)]="searchQuery" (keyup.enter)="loadRecords()" placeholder="Search name, email, phone...">
    <button class="btn btn-outline" (click)="loadRecords()">🔍 Search</button>
    <button class="btn btn-outline" (click)="loadRecords()">🔄</button>
  </div>

  @if (toastMsg) {<div class="toast">{{ toastMsg }}</div>}

  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Date</th>
        <th>First Name</th>
        <th>Last Name</th>
        <th>Email</th>
        <th>SSN</th>
        <th>Phone</th>
        <th>Bank</th>
        <th>State</th>
        <th>Actions</th>
      </tr>
    </thead>
    <tbody>
      @for (r of records(); track r.id) {
        <tr>
          <td>{{ r.id }}</td>
          <td class="date">{{ r.created_at.substring(0,10) }}</td>
          <td>{{ r.first_name }}</td>
          <td>{{ r.last_name }}</td>
          <td class="email">{{ r.email }}</td>
          <td>{{ r.ssn }}</td>
          <td>{{ r.phone }}</td>
          <td>{{ r.bank_name }}</td>
          <td>{{ r.state }}</td>
          <td class="actions">
            <button class="btn-sm" (click)="editRecord(r)">✏️</button>
            <button class="btn-sm danger" (click)="confirmDelete(r)">🗑️</button>
          </td>
        </tr>
      } @empty {
        <tr><td colspan="10" class="empty">No records yet. Click "New OCR" to start.</td></tr>
      }
    </tbody>
  </table>

  <!-- Edit Modal -->
  @if (editRec) {
    <div class="modal-overlay" (click)="closeEdit()">
      <div class="modal" (click)="$event.stopPropagation()">
        <div class="modal-header">
          <h2>{{ editRec.id ? 'Record #' + editRec.id : 'New Record' }}</h2>
          <button class="btn-sm" (click)="closeEdit()">✕</button>
        </div>
        <div class="modal-body">
          <div class="field-grid">
            @for (f of fieldDefs; track f.key) {
              <div class="field">
                <label>{{ f.label }}</label>
                <input [value]="getField(f.key)" (input)="setField(f.key, $event)">
              </div>
            }
          </div>
          @if (editRec.raw_ocr) {
            <details><summary>Raw OCR</summary><pre>{{ editRec.raw_ocr }}</pre></details>
          }
        </div>
        <div class="modal-footer">
          <button class="btn btn-outline" (click)="closeEdit()">Cancel</button>
          @if (editRec.id) {<button class="btn btn-danger" (click)="deleteRecord(editRec)">Delete</button>}
          <button class="btn btn-primary" (click)="saveRecord()">💾 Save</button>
        </div>
      </div>
    </div>
  }

  <!-- Delete Confirm -->
  @if (delRec) {
    <div class="modal-overlay" (click)="delRec = null">
      <div class="modal modal-sm" (click)="$event.stopPropagation()">
        <p>Delete record #{{ delRec.id }} ({{ delRec.first_name }} {{ delRec.last_name }})?</p>
        <div class="modal-footer">
          <button class="btn btn-outline" (click)="delRec = null">Cancel</button>
          <button class="btn btn-danger" (click)="deleteRecord(delRec)">Delete</button>
        </div>
      </div>
    </div>
  }
</div>
  `,
  styles: [`
:host { display:block; background:#0f1117; color:#e1e4ed; min-height:100dvh; font-family:Inter,sans-serif; }
.app { max-width:1200px; margin:0 auto; padding:20px; }

header { display:flex; justify-content:space-between; align-items:center; margin-bottom:20px; }
header h1 { font-size:1.4rem; font-weight:600; }
.header-info { display:flex; gap:12px; align-items:center; }
.badge { padding:3px 10px; border-radius:20px; font-size:0.75rem; background:#242837; color:#8b90a3; }
.badge.busy { background:#4f8cff; color:#fff; }
.count { font-size:0.85rem; color:#8b90a3; }

.toolbar { display:flex; gap:8px; margin-bottom:16px; align-items:center; }
.search { flex:1; padding:8px 12px; border-radius:8px; border:1px solid #2d3248; background:#1a1d27; color:#e1e4ed; font-size:0.85rem; outline:none; }
.search:focus { border-color:#4f8cff; }

.btn { padding:8px 16px; border-radius:8px; border:none; font-size:0.85rem; cursor:pointer; transition:.15s; white-space:nowrap; }
.btn:disabled { opacity:0.5; cursor:not-allowed; }
.btn-primary { background:#4f8cff; color:#fff; }
.btn-primary:hover:not(:disabled) { background:#3a73e0; }
.btn-outline { background:transparent; color:#8b90a3; border:1px solid #2d3248; }
.btn-outline:hover { background:#242837; }
.btn-danger { background:#f87171; color:#fff; }
.btn-danger:hover { background:#e05050; }
.btn-sm { padding:4px 8px; border-radius:6px; border:1px solid #2d3248; background:transparent; color:#8b90a3; cursor:pointer; font-size:0.8rem; }
.btn-sm:hover { background:#242837; color:#e1e4ed; }
.btn-sm.danger:hover { color:#f87171; border-color:#f87171; }

.toast { padding:10px 16px; background:#1a1d27; border:1px solid #4f8cff; border-radius:8px; margin-bottom:12px; font-size:0.85rem; }

table { width:100%; border-collapse:collapse; font-size:0.85rem; }
th { text-align:left; padding:8px 10px; color:#8b90a3; font-weight:500; border-bottom:1px solid #2d3248; white-space:nowrap; }
td { padding:8px 10px; border-bottom:1px solid #1a1d27; }
tr:hover td { background:#1a1d27; }
td.email { max-width:180px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
td.date { color:#8b90a3; }
td.actions { white-space:nowrap; }
.empty { text-align:center; color:#8b90a3; padding:40px; }

/* Modal */
.modal-overlay { position:fixed; inset:0; background:rgba(0,0,0,0.6); display:flex; align-items:center; justify-content:center; z-index:100; }
.modal { background:#1a1d27; border:1px solid #2d3248; border-radius:12px; width:700px; max-width:95vw; max-height:90vh; overflow-y:auto; }
.modal-sm { width:400px; padding:24px; }
.modal-header { display:flex; justify-content:space-between; align-items:center; padding:16px 20px; border-bottom:1px solid #2d3248; }
.modal-header h2 { font-size:1.1rem; font-weight:600; }
.modal-body { padding:16px 20px; }
.modal-footer { display:flex; gap:8px; justify-content:flex-end; padding:12px 20px; border-top:1px solid #2d3248; }

.field-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.field { display:flex; flex-direction:column; gap:4px; }
.field label { font-size:0.7rem; color:#8b90a3; text-transform:uppercase; letter-spacing:0.5px; }
.field input { padding:8px 10px; border-radius:6px; border:1px solid #2d3248; background:#242837; color:#e1e4ed; font-size:0.85rem; outline:none; }
.field input:focus { border-color:#4f8cff; }

details { margin-top:12px; }
details summary { cursor:pointer; font-size:0.8rem; color:#8b90a3; }
details pre { margin-top:8px; padding:10px; background:#0f1117; border-radius:6px; font-size:0.75rem; white-space:pre-wrap; word-break:break-all; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#2d3248; border-radius:3px; }
  `]
})
export class App implements OnInit, OnDestroy {
  api = inject(ApiService);
  status = this.api.status;
  records = signal<Record[]>([]);
  totalRecords = 0;
  searchQuery = '';
  toastMsg = '';
  editRec: Record | null = null;
  delRec: Record | null = null;
  pollingTimer: any = null;

  fieldDefs = [
    { key:'first_name', label:'First Name' }, { key:'last_name', label:'Last Name' },
    { key:'email', label:'Email' }, { key:'ssn', label:'SSN' },
    { key:'phone', label:'Phone' }, { key:'bank_name', label:'Bank Name' },
    { key:'account_no', label:'A/C No' }, { key:'loan_amount', label:'Loan Amount' },
    { key:'address', label:'Address' }, { key:'city', label:'City' },
    { key:'state', label:'State' }, { key:'zip', label:'Zip' },
    { key:'dob', label:'DOB' }, { key:'licence_no', label:'Licence No' },
    { key:'licence_state', label:'Licence State' }, { key:'ip', label:'IP' },
  ];

  ngOnInit() {
    this.loadRecords();
    this.pollingTimer = setInterval(() => this.api.refreshStatus(), 3000);
  }
  ngOnDestroy() { if (this.pollingTimer) clearInterval(this.pollingTimer); }

  async triggerOcr() {
    await this.api.triggerOcr();
    this.showToast('OCR triggered — fields being filled...');
    // Wait then refresh
    setTimeout(() => this.loadRecords(), 3000);
  }

  async loadRecords() {
    const url = this.searchQuery
      ? `/api/records?search=${encodeURIComponent(this.searchQuery)}&limit=100`
      : '/api/records?limit=100';
    try {
      const r = await fetch(url);
      const data = await r.json();
      this.records.set(data.records);
      this.totalRecords = data.total;
    } catch {}
  }

  editRecord(r: Record) {
    this.editRec = { ...r };
  }

  closeEdit() {
    this.editRec = null;
  }

  async saveRecord() {
    if (!this.editRec) return;
    const isNew = !this.editRec.id;
    const url = isNew ? '/api/records' : `/api/records/${this.editRec.id}`;
    const method = isNew ? 'POST' : 'PUT';
    try {
      await fetch(url, { method, headers:{'Content-Type':'application/json'}, body: JSON.stringify(this.editRec) });
      this.showToast(isNew ? 'Record created' : 'Record saved');
      this.editRec = null;
      this.loadRecords();
    } catch (e) {
      this.showToast('Save failed');
    }
  }

  confirmDelete(r: Record) {
    this.delRec = r;
  }

  async deleteRecord(r: Record) {
    try {
      await fetch(`/api/records/${r.id}`, { method:'DELETE' });
      this.showToast('Record deleted');
      if (this.editRec?.id === r.id) this.editRec = null;
      this.delRec = null;
      this.loadRecords();
    } catch {
      this.showToast('Delete failed');
    }
  }

  getField(key: string): string {
    return this.editRec ? (this.editRec as any)[key] || '' : '';
  }

  setField(key: string, event: Event) {
    if (this.editRec) {
      (this.editRec as any)[key] = (event.target as HTMLInputElement).value;
    }
  }

  showToast(msg: string) {
    this.toastMsg = msg;
    setTimeout(() => this.toastMsg = '', 3000);
  }
}
