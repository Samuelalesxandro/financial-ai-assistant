
import os
import re
import json
import argparse

# Dependências externas (do requirements.txt)
from dotenv import load_dotenv
from openai import OpenAI
from gtts import gTTS


# -----------------------------
# Utilidades de arquivo/KB
# -----------------------------
def load_knowledge_base(path: str) -> dict:
    """Carrega base de conhecimento simples (FAQ) em JSON."""
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def retrieve_from_kb(kb: dict, user_text: str):
    """
    Busca simples: se alguma chave do KB aparece como substring no texto do usuário, retorna a resposta.
    """
    if not kb:
        return None

    text = user_text.lower()
    # chaves mais longas primeiro (melhora match)
    for key in sorted(kb.keys(), key=len, reverse=True):
        if key.lower() in text:
            return kb[key]
    return None


# -----------------------------
# Simulações financeiras simples
# -----------------------------
def saldo_diario(linha: str) -> float:
    """
    Entrada tipo: "R 100.00,D 50.00,R 20.00"
    Saldo = receitas - despesas
    """
    saldo = 0.0
    lancamentos = [p.strip() for p in linha.split(",") if p.strip()]
    for l in lancamentos:
        partes = l.split()
        if len(partes) != 2:
            continue
        tipo, valor_str = partes
        valor = float(valor_str)
        if tipo.upper() == "R":
            saldo += valor
        elif tipo.upper() == "D":
            saldo -= valor
    return saldo


def juros_simples(principal: float, taxa_mensal: float, meses: int) -> float:
    """
    Montante com juros simples: M = P * (1 + i*t)
    taxa_mensal em decimal (ex.: 0.02 = 2%)
    """
    return principal * (1 + taxa_mensal * meses)


def juros_compostos(principal: float, taxa_mensal: float, meses: int) -> float:
    """
    Montante com juros compostos: M = P * (1 + i)^t
    """
    return principal * ((1 + taxa_mensal) ** meses)


def parcela_price(principal: float, taxa_mensal: float, meses: int) -> float:
    """
    Parcela no sistema Price:
    PMT = P * i / (1 - (1+i)^-n)
    """
    if meses <= 0:
        return 0.0
    if taxa_mensal == 0:
        return principal / meses
    i = taxa_mensal
    return principal * (i / (1 - (1 + i) ** (-meses)))


def detectar_simulacao(user_text: str):
    """
    Detecta se o usuário quer uma simulação e tenta extrair parâmetros.
    Retorna (tipo, resultado_texto) ou (None, None).
    """
    t = user_text.lower()

    # 1) saldo do dia -> procura padrão "R 100,D 50" ou menção "saldo" + "R/D"
    if ("saldo" in t) and (("r " in t) or ("d " in t) or ("," in t)):
        # tenta achar uma sequência com R/D e valores
        # ex: "R 100.00,D 50.00,R 20.00"
        m = re.search(r"([rdRD]\s*\d+(\.\d+)?\s*(,\s*[rdRD]\s*\d+(\.\d+)?\s*)+)", user_text)
        if m:
            linha = m.group(1)
            s = saldo_diario(linha)
            return ("saldo", f"Saldo do dia (receitas - despesas): {s:.2f}\n"
                             f"✅ Cálculo demonstrativo: somei 'R' e subtraí 'D'.")

    # 2) juros simples/compostos -> extrai principal, taxa (% a.m.) e meses
    # Ex: "2.000 a 2% ao mês por 12 meses"
    if "juros" in t or "emprest" in t or "parcela" in t:
        # principal: tenta achar primeiro número grande
        p = re.search(r"(\d{1,3}(\.\d{3})+|\d+)(,\d+)?", t)  # 2000 ou 2.000 ou 2.000,50
        principal = None
        if p:
            raw = p.group(0)
            raw = raw.replace(".", "").replace(",", ".")
            try:
                principal = float(raw)
            except:
                principal = None

        # taxa: "2%" ou "2 %"
        rtax = re.search(r"(\d+(\.\d+)?)\s*%\s*(ao\s*m[eê]s|a\.?\s*m\.?)?", t)
        taxa = None
        if rtax:
            taxa = float(rtax.group(1)) / 100.0

        # meses: "por 12 meses" ou "12 meses"
        rmes = re.search(r"(\d+)\s*mes", t)
        meses = None
        if rmes:
            meses = int(rmes.group(1))

        # se tiver principal+taxa+meses
        if principal is not None and taxa is not None and meses is not None:
            # se pedir parcela -> Price
            if "parcela" in t or "price" in t:
                pmt = parcela_price(principal, taxa, meses)
                return ("parcela", f"Parcela (Price) estimada: {pmt:.2f}\n"
                                   f"📌 Parâmetros: principal={principal:.2f}, taxa={taxa*100:.2f}% a.m., meses={meses}\n"
                                   f"✅ Cálculo demonstrativo (Price).")

            # se mencionar compostos
            if "compost" in t:
                mnt = juros_compostos(principal, taxa, meses)
                return ("juros_compostos", f"Montante (juros compostos): {mnt:.2f}\n"
                                           f"📌 Parâmetros: principal={principal:.2f}, taxa={taxa*100:.2f}% a.m., meses={meses}\n"
                                           f"✅ Fórmula: M = P(1+i)^t")

            # padrão: simples
            mnt = juros_simples(principal, taxa, meses)
            return ("juros_simples", f"Montante (juros simples): {mnt:.2f}\n"
                                     f"📌 Parâmetros: principal={principal:.2f}, taxa={taxa*100:.2f}% a.m., meses={meses}\n"
                                     f"✅ Fórmula: M = P(1+i·t)")

    return (None, None)


# -----------------------------
# OpenAI: STT e Chat
# -----------------------------
def transcrever_audio(client: OpenAI, audio_path: str, model_stt: str) -> str:
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model=model_stt,
            file=audio_file
        )
    return transcription.text


def responder_com_ia(client: OpenAI, model_chat: str, system_prompt: str, messages: list) -> str:
    resp = client.chat.completions.create(
        model=model_chat,
        messages=[{"role": "system", "content": system_prompt}] + messages
    )
    return resp.choices[0].message.content


def sintetizar_tts(texto: str, lang: str, output_mp3: str):
    tts = gTTS(text=texto, lang=lang, slow=False)
    tts.save(output_mp3)


# -----------------------------
# App principal (CLI)
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Agente financeiro com IA (FAQ + simulações + UX), com opção de voz (STT/TTS).")
    parser.add_argument("--kb", default="data/knowledge_base.json", help="Caminho do JSON da base de conhecimento (FAQ).")
    parser.add_argument("--model-chat", default="gpt-4o-mini", help="Modelo do Chat (ex.: gpt-4o-mini).")
    parser.add_argument("--model-stt", default="whisper-1", help="Modelo de transcrição (ex.: whisper-1).")
    parser.add_argument("--audio", default=None, help="(Opcional) Caminho de um arquivo de áudio para transcrição e pergunta.")
    parser.add_argument("--out", default="resposta.mp3", help="(Opcional) Nome do MP3 de saída (TTS).")
    parser.add_argument("--lang", default="pt", help="Idioma do TTS (gTTS). Ex.: pt, en, es.")
    parser.add_argument("--no-tts", action="store_true", help="Desliga a geração de áudio de resposta.")
    parser.add_argument("--max-turns", type=int, default=6, help="Quantas mensagens manter no contexto (memória curta).")
    args = parser.parse_args()

    load_dotenv()  # carrega .env se existir
    api_key = os.getenv("OPENAI_API_KEY")

    kb = load_knowledge_base(args.kb)

    # Prompt de segurança/UX (alinhado ao desafio)
    system_prompt = (
        "Você é um assistente financeiro educacional, claro e didático.\n"
        "Siga boas práticas de UX: responda com objetividade, passo a passo quando houver cálculo, e linguagem simples.\n"
        "Não solicite dados sensíveis (senha, token, documento). Não ofereça aconselhamento financeiro personalizado.\n"
        "Se a pergunta envolver decisão financeira, inclua um aviso de que é conteúdo educativo.\n"
        "Se houver incerteza, recomende confirmar em canais oficiais."
    )

    # Memória curta (somente durante a execução)
    history = []

    # Se o usuário passar um áudio, transcreve e usa como pergunta inicial
    if args.audio:
        if not api_key:
            raise RuntimeError("Para usar --audio (transcrição), defina OPENAI_API_KEY no ambiente (.env ou variável).")
        client = OpenAI(api_key=api_key)
        print("🔊 Transcrevendo áudio...")
        user_text = transcrever_audio(client, args.audio, args.model_stt)
        print("\n📝 Transcrição:")
        print(user_text)
        print("\n---\n")
        # processa 1 interação e encerra (modo arquivo)
        resposta_final = processar_interacao(
            api_key=api_key,
            kb=kb,
            system_prompt=system_prompt,
            model_chat=args.model_chat,
            user_text=user_text,
            history=history,
            max_turns=args.max_turns,
        )
        print("✅ Resposta:")
        print(resposta_final)

        if not args.no_tts:
            print("\n🗣️ Gerando áudio (TTS)...")
            sintetizar_tts(resposta_final, args.lang, args.out)
            print(f"🎧 Áudio salvo em: {args.out}")
        return

    # Modo chat via terminal (texto)
    print("🏦 Agente Financeiro com IA (modo texto)")
    print("Digite sua pergunta. Para sair, digite: sair\n")

    while True:
        try:
            user_text = input("Você: ").strip()
        except EOFError:
            break

        if not user_text:
            print("Assistente: Pode escrever sua dúvida (ex.: 'o que é juros compostos?').\n")
            continue

        if user_text.lower() in ("sair", "exit", "quit"):
            print("Assistente: Até mais! 👋")
            break

        resposta_final = processar_interacao(
            api_key=api_key,
            kb=kb,
            system_prompt=system_prompt,
            model_chat=args.model_chat,
            user_text=user_text,
            history=history,
            max_turns=args.max_turns,
        )

        print("\nAssistente:", resposta_final, "\n")

        if (not args.no_tts) and resposta_final and len(resposta_final) > 0:
            try:
                sintetizar_tts(resposta_final, args.lang, args.out)
                print(f"🎧 (TTS) Áudio atualizado em: {args.out}\n")
            except Exception:
                # não falha a aplicação caso TTS tenha problema
                pass


def processar_interacao(api_key, kb, system_prompt, model_chat, user_text, history, max_turns):
    """
    1) tenta simulações locais
    2) tenta FAQ na base de conhecimento
    3) se tiver OPENAI_API_KEY, usa LLM
    4) senão, retorna fallback com orientação
    """
    # 1) simulação local
    sim_tipo, sim_resp = detectar_simulacao(user_text)
    if sim_resp:
        # guarda na memória curta
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": sim_resp})
        del history[:-max_turns]
        return sim_resp

    # 2) FAQ (KB)
    kb_resp = retrieve_from_kb(kb, user_text)
    if kb_resp:
        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": kb_resp})
        del history[:-max_turns]
        return kb_resp

    # 3) LLM (se tiver chave)
    if api_key:
        client = OpenAI(api_key=api_key)

        # monta contexto (memória curta)
        messages = history[-max_turns:] + [{"role": "user", "content": user_text}]

        resposta = responder_com_ia(client, model_chat, system_prompt, messages)

        # inclui aviso padrão (educacional) quando necessário
        aviso = ""
        if any(p in user_text.lower() for p in ["invest", "aplicar", "comprar", "vender", "melhor", "recomenda"]):
            aviso = "\n\n⚠️ Nota: Conteúdo educacional. Para decisões financeiras, confirme condições e riscos em canais oficiais ou com profissional."

        resposta_final = (resposta.strip() + aviso).strip()

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": resposta_final})
        del history[:-max_turns]
        return resposta_final

    # 4) fallback sem LLM
    fallback = (
        "Eu consigo ajudar com FAQs e simulações simples, mas para respostas geradas por IA você precisa configurar a variável OPENAI_API_KEY.\n"
        "Exemplos do que posso calcular sem IA:\n"
        "- Saldo do dia: 'saldo R 100.00,D 50.00,R 20.00'\n"
        "- Juros: '2.000 a 2% ao mês por 12 meses (simples/compostos)'\n"
        "- Parcela: 'parcela de 2000 a 2% por 12 meses'\n"
    )
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": fallback})
    del history[:-max_turns]
    return fallback


if __name__ == "__main__":
    main()
