// Types and fetch helpers for the FastAPI backend (docs/SPEC.md §6 and §8).

export interface ContractSummary {
  id: number;
  title: string;
  split: "dev" | "test";
}

export interface Contract extends ContractSummary {
  text: string;
}

export interface Citation {
  chunk_id: number;
  contract_id: number;
  start_char: number; // offsets into Contract.text
  end_char: number;
  quote: string;
}

export interface Claim {
  text: string;
  citations: Citation[];
}

export interface QueryResponse {
  answer: string;
  claims: Claim[];
  refused: boolean;
  dropped_claims: number;
  trace_id: string;
}

async function getJson<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail ?? detail;
    } catch {
      // body wasn't JSON; keep the status text
    }
    throw new Error(`${res.status}: ${typeof detail === "string" ? detail : JSON.stringify(detail)}`);
  }
  return res.json() as Promise<T>;
}

export const listContracts = () => getJson<ContractSummary[]>("/contracts");

export const getContract = (id: number) => getJson<Contract>(`/contracts/${id}`);

export const query = (question: string, contract_id: number | null) =>
  getJson<QueryResponse>("/query", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, contract_id }),
  });
