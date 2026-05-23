import { Injectable, signal } from '@angular/core';

export interface Status {
  busy: boolean;
  last_ocr: string;
  last_fields: string[];
  memory_count: number;
  neural_trained: number;
  neural_confident: boolean;
}

export interface LogEntry {
  level: string;
  message: string;
}

@Injectable({ providedIn: 'root' })
export class ApiService {
  private base = '';

  status = signal<Status | null>(null);
  logs = signal<LogEntry[]>([]);
  connected = signal(false);

  private ws: WebSocket | null = null;

  async refreshStatus() {
    try {
      const r = await fetch(`${this.base}/api/status`);
      this.status.set(await r.json());
    } catch { /* ignore */ }
  }

  async getConfig(): Promise<any> {
    try {
      const r = await fetch(`${this.base}/api/config`);
      return await r.json();
    } catch { return null; }
  }

  async triggerOcr() {
    await fetch(`${this.base}/api/trigger`, { method: 'POST' });
  }

  async triggerLearn() {
    await fetch(`${this.base}/api/learn`, { method: 'POST' });
  }

  connectLogs() {
    if (this.ws) return;
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const host = location.host;
    const url = `${proto}://${host}/ws/logs`;
    this.ws = new WebSocket(url);
    this.ws.onopen = () => this.connected.set(true);
    this.ws.onclose = () => { this.connected.set(false); this.ws = null; };
    this.ws.onerror = () => { this.connected.set(false); };
    this.ws.onmessage = (ev) => {
      try {
        const data = JSON.parse(ev.data) as LogEntry;
        this.logs.update(l => [...l.slice(-499), data]);
      } catch { /* ignore */ }
    };
  }
}
