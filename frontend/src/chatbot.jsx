import React, { useState, useRef, useEffect } from 'react';

export default function Chatbot() {
  const [messages, setMessages] = useState([
    {
      id: 0,
      type: 'bot',
      content: "Hello! I'm your CCTV security consultant. To get started, please upload a floor plan of your property using the camera button below — this helps me give you accurate camera placement recommendations.",
      timestamp: new Date(),
    },
  ]);

  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [requirements, setRequirements] = useState(null);
  const [stage, setStage] = useState('gathering');
  const [conversationHistory, setConversationHistory] = useState([]);
  const [floorPlan, setFloorPlan] = useState(null);
  const [summary, setSummary] = useState(null);
  const [progress, setProgress] = useState(0);
  const [decisionLoading, setDecisionLoading] = useState(false);
  const summaryFetched = useRef(false);
  const conversationComplete = useRef(false); // stays true once stage hits complete

  // Firebase session state
  const [projectId, setProjectId] = useState(null);
  const [conversationId, setConversationId] = useState(null);
  const [sessionReady, setSessionReady] = useState(false);

  const messagesEndRef = useRef(null);
  const abortControllerRef = useRef(null);
  const floorPlanInputRef = useRef(null);
  const sessionInitialized = useRef(false); // ← StrictMode double-fire guard

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    return () => { abortControllerRef.current?.abort(); };
  }, []);

  // Create project + conversation once on mount (guarded against StrictMode double-invoke)
  useEffect(() => {
    if (sessionInitialized.current) return;
    sessionInitialized.current = true;

    const initSession = async () => {
      try {
        const backendURL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

        const projectRes = await fetch(`${backendURL}/projects`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name: `CCTV Project — ${new Date().toLocaleDateString()}`,
            client_email: 'guest@example.com',
          }),
        });

        if (!projectRes.ok) {
          console.warn('Could not create project — Firebase saving disabled.');
          setSessionReady(true);
          return;
        }

        const projectData = await projectRes.json();
        const newProjectId = projectData.project_id;
        setProjectId(newProjectId);

        const convRes = await fetch(`${backendURL}/projects/${newProjectId}/conversations`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
        });

        if (!convRes.ok) {
          console.warn('Could not create conversation — Firebase saving disabled.');
          setSessionReady(true);
          return;
        }

        const convData = await convRes.json();
        setConversationId(convData.conversation_id);
        console.log(`✓ Session ready — project: ${newProjectId}, conversation: ${convData.conversation_id}`);
        setSessionReady(true);

      } catch (err) {
        console.warn('Firebase session init failed (backend may be offline):', err);
        setSessionReady(true);
      }
    };

    initSession();
  }, []);

  const addMessage = (type, content) => {
    setMessages((prev) => [...prev, {
      id: prev.length,
      type,
      content,
      timestamp: new Date(),
    }]);
  };

  const handleFloorPlanUpload = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const allowedTypes = ['image/jpeg', 'image/png', 'image/gif', 'image/webp'];
    if (!allowedTypes.includes(file.type)) {
      addMessage('bot', 'Please upload an image file (JPG, PNG, GIF, or WEBP) of your floor plan.');
      e.target.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = async (event) => {
      const base64Data = event.target.result;
      const floorPlanData = { name: file.name, type: file.type, data: base64Data };
      setFloorPlan(floorPlanData);

      setMessages((prev) => [...prev, {
        id: prev.length,
        type: 'image',
        content: base64Data,
        fileName: file.name,
        timestamp: new Date(),
      }]);

      abortControllerRef.current?.abort();
      const controller = new AbortController();
      abortControllerRef.current = controller;
      setIsLoading(true);

      const analyzeMessage = `[Floor plan uploaded: ${file.name}] Please analyse this floor plan for CCTV coverage.`;
      const response = await callBackendAPI(analyzeMessage, conversationHistory, controller.signal, floorPlanData);

      if (response === 'aborted') { setIsLoading(false); return; }

      if (!response) {
        addMessage('bot', "Sorry, I couldn't analyse the floor plan. Please check the backend is running and try again.");
        setIsLoading(false);
        return;
      }

      handleResponse(response, analyzeMessage, conversationHistory);
      setIsLoading(false);
    };

    reader.readAsDataURL(file);
    e.target.value = '';
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    abortControllerRef.current?.abort();
    const controller = new AbortController();
    abortControllerRef.current = controller;

    const userMessage = input;
    addMessage('user', userMessage);
    setInput('');
    setIsLoading(true);

    const response = await callBackendAPI(userMessage, conversationHistory, controller.signal, null);

    if (response === 'aborted') { setIsLoading(false); return; }

    if (!response) {
      addMessage('bot', "Sorry, I'm having trouble connecting to the backend. Please check that the server is running on http://localhost:8000");
      setConversationHistory((prev) => [...prev, { role: 'user', content: userMessage }]);
      setIsLoading(false);
      return;
    }

    await handleResponse(response, userMessage, conversationHistory);
    setIsLoading(false);
  };

  const handleResponse = async (response, userMessage, history) => {
    if (!response) return;

    addMessage('bot', response.bot_response);

    const freshRequirements =
      response.requirements && Object.keys(response.requirements).length > 0
        ? response.requirements : null;

    if (freshRequirements) setRequirements(freshRequirements);
    if (response.stage) setStage(response.stage);
    if (typeof response.progress === 'number') setProgress(response.progress);

    const updatedHistory = [
      ...history,
      { role: 'user', content: userMessage },
      { role: 'assistant', content: response.bot_response },
    ];
    setConversationHistory(updatedHistory);

    // Summary comes inline from /chat when stage === 'complete'
    if (response.summary && !summaryFetched.current) {
      summaryFetched.current = true;
      setSummary(response.summary);
      setMessages((prev) => [...prev, {
        id: prev.length, 
        type: 'summary', 
        content: response.summary, 
        timestamp: new Date()
      }]);

      await fetchDecision(response.summary)
    }
  };

  const fetchDecision = async (summaryText) => {
    setDecisionLoading(true);
    try {
      const backendURL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';
      console.log('[fetchDecision] POST', `${backendURL}/decision`);
      
      const res = await fetch(`${backendURL}/decision`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          requirement: summaryText
        }),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        console.error('[fetchDecision] Server error:', res.status, err);
        setMessages((prev) => [...prev, {
          id: prev.length,
          type: 'bot',
          content: "I had trouble generating the final decision based on the summary.",
          timestamp: new Date(),
        }]);
        setDecisionLoading(false);
        return;
      }

      const data = await res.json();
      // Assuming the backend returns the markdown text inside `decision`
      const decisionContent = data.decision || data.output; 

      console.log('[fetchDecision] Received decision content:', decisionContent);

      if (decisionContent) {
        setMessages((prev) => [...prev, {
          id: prev.length,
          type: 'decision',
          content: decisionContent,
          timestamp: new Date(),
        }]);
      }
    } catch (err) {
      console.error('[fetchDecision] Network/fetch error:', err);
      setMessages((prev) => [...prev, {
        id: prev.length,
        type: 'bot',
        content: "Sorry, I had trouble reaching the decision endpoint.",
        timestamp: new Date(),
      }]);
    }
    setDecisionLoading(false);
  };

  const callBackendAPI = async (message, history, signal, floorPlanData) => {
    try {
      const backendURL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:8000';

      const body = {
        message,
        conversation_history: history
      };

      if (projectId) body.project_id = projectId;
      if (conversationId) body.conversation_id = conversationId;
      // Pass last known requirements so backend can keep stage stable on skipped extraction turns
      if (requirements) body.cached_requirements = requirements;

      if (floorPlanData) {
        body.floor_plan = {
          name: floorPlanData.name,
          type: floorPlanData.type,
          data: floorPlanData.data,
        };
      }

      const response = await fetch(`${backendURL}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal,
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        console.error('Backend error:', response.status, errorData);
        return null;
      }

      return await response.json();
    } catch (error) {
      if (error.name === 'AbortError') return 'aborted';
      console.error('Error calling backend:', error);
      return null;
    }
  };



  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const getStageBadge = () => {
    const stages = {
      gathering:  { label: 'Gathering Info',      color: '#3B82F6' },
      clarifying: { label: 'Clarifying Details',  color: '#F59E0B' },
      complete:   { label: 'Complete',             color: '#10B981' },
    };
    return stages[stage] || stages.gathering;
  };

  const badge = getStageBadge();

  // Markdown renderer for summary messages
  const renderMarkdown = (text) =>
    text.split('\n').map((line, i) => {
      if (/^### /.test(line))
        return (
          <div key={i} style={{ fontWeight: 700, fontSize: '12px', marginTop: '0.75rem', marginBottom: '2px', color: 'var(--color-text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            {line.replace(/^### /, '')}
          </div>
        );
      if (/^## /.test(line))
        return (
          <div key={i} style={{ fontWeight: 700, fontSize: '14px', marginTop: '1rem', marginBottom: '4px', color: 'var(--color-text-primary)' }}>
            {line.replace(/^## /, '')}
          </div>
        );
      if (/^- /.test(line)) {
        const content = line.replace(/^- /, '').replace(/\*\*(.*?)\*\*/g, (_, m) => `__BOLD__${m}__BOLD__`);
        const parts = content.split('__BOLD__');
        return (
          <div key={i} style={{ display: 'flex', gap: '6px', marginBottom: '2px' }}>
            <span style={{ color: 'var(--color-text-secondary)', flexShrink: 0 }}>•</span>
            <span>{parts.map((p, j) => j % 2 === 1 ? <strong key={j}>{p}</strong> : p)}</span>
          </div>
        );
      }
      if (line.trim() === '') return <div key={i} style={{ height: '4px' }} />;
      return <div key={i}>{line}</div>;
    });

  return (
    <div style={{
      background: 'var(--color-background-primary)',
      borderRadius: 'var(--border-radius-lg)',
      border: '0.5px solid var(--color-border-tertiary)',
      display: 'flex',
      flexDirection: 'column',
      height: '650px',
      maxWidth: '680px',
      margin: '0 auto',
      overflow: 'hidden',
    }}>
      {/* Header */}
      <div style={{
        padding: '1rem',
        borderBottom: '0.5px solid var(--color-border-tertiary)',
        background: 'var(--color-background-secondary)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
          <h2 style={{ margin: 0, fontSize: '16px', fontWeight: 500, color: 'var(--color-text-primary)' }}>
            CCTV Requirements Chatbot
          </h2>
          {/* Firebase status */}
          {sessionReady && (
            <span style={{ fontSize: '11px', color: projectId ? '#10B981' : '#F59E0B' }}>
              {projectId ? '● Saving' : '● Not saving'}
            </span>
          )}
        </div>

        {/* Stage badge + progress bar */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            display: 'inline-block', width: '8px', height: '8px',
            borderRadius: '50%', background: badge.color, flexShrink: 0,
          }} />
          <span style={{ fontSize: '12px', color: 'var(--color-text-secondary)', flexShrink: 0 }}>
            {badge.label}
          </span>
          {/* Progress bar */}
          <div style={{ flex: 1, height: '4px', background: 'var(--color-border-tertiary)', borderRadius: '2px', overflow: 'hidden' }}>
            <div style={{
              height: '100%',
              width: `${progress}%`,
              background: badge.color,
              borderRadius: '2px',
              transition: 'width 0.5s ease',
            }} />
          </div>
          <span style={{ fontSize: '11px', color: 'var(--color-text-secondary)', flexShrink: 0 }}>
            {progress}%
          </span>
        </div>
      </div>

      {/* Messages */}
      <div style={{
        flex: 1, overflowY: 'auto', padding: '1rem',
        display: 'flex', flexDirection: 'column', gap: '0.75rem',
      }}>
        {messages.map((msg) => {
          if (msg.type === 'image') {
            return (
              <div key={msg.id} style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <div style={{
                  maxWidth: '75%', borderRadius: 'var(--border-radius-md)',
                  overflow: 'hidden', border: '1px solid var(--color-border-tertiary)',
                }}>
                  <img
                    src={msg.content} alt={msg.fileName}
                    style={{ display: 'block', maxWidth: '100%', maxHeight: '260px', objectFit: 'contain' }}
                  />
                  <div style={{
                    padding: '4px 10px', fontSize: '11px',
                    color: 'var(--color-text-secondary)', background: 'var(--color-background-secondary)',
                  }}>
                    {msg.fileName}
                  </div>
                </div>
              </div>
            );
          }

          if (msg.type === 'summary') {
            return (
              <div key={msg.id} style={{ width: '100%' }}>
                <div style={{
                  border: '1px solid var(--color-border-tertiary)',
                  borderRadius: 'var(--border-radius-md)', overflow: 'hidden',
                }}>
                  <div style={{
                    padding: '0.6rem 1rem', background: 'var(--color-background-info)',
                    color: 'var(--color-text-info)', fontSize: '13px', fontWeight: 600, letterSpacing: '0.02em',
                  }}>
                    Requirements Summary
                  </div>
                  <div style={{
                    padding: '1rem', background: 'var(--color-background-secondary)',
                    fontSize: '13px', lineHeight: '1.8', color: 'var(--color-text-primary)', wordBreak: 'break-word',
                  }}>
                    {renderMarkdown(msg.content)}
                  </div>
                </div>
              </div>
            );
          }

          if (msg.type === 'decision') {
            return (
              <div key={msg.id} style={{ width: '100%' }}>
                <div style={{
                  border: '1px solid var(--color-border-tertiary)',
                  borderRadius: 'var(--border-radius-md)', overflow: 'hidden',
                }}>
                  <div style={{
                    padding: '0.6rem 1rem', 
                    // Differentiate the header color slightly for the decision
                    background: '#10B98120',
                    color: '#10B981', 
                    fontSize: '13px', fontWeight: 600, letterSpacing: '0.02em',
                  }}>
                    {'System Decision'}
                  </div>
                  <div style={{
                    padding: '1rem', background: 'var(--color-background-secondary)',
                    fontSize: '13px', lineHeight: '1.8', color: 'var(--color-text-primary)', wordBreak: 'break-word',
                  }}>
                    {renderMarkdown(msg.content)}
                  </div>
                </div>
              </div>
            );
          }

          return (
            <div key={msg.id} style={{ display: 'flex', justifyContent: msg.type === 'bot' ? 'flex-start' : 'flex-end' }}>
              <div style={{
                maxWidth: '75%', padding: '0.75rem 1rem',
                borderRadius: 'var(--border-radius-md)', fontSize: '14px', lineHeight: '1.6',
                background: msg.type === 'bot' ? 'var(--color-background-secondary)' : 'var(--color-background-info)',
                color: msg.type === 'bot' ? 'var(--color-text-primary)' : 'var(--color-text-info)',
                wordBreak: 'break-word',
              }}>
                {msg.content}
              </div>
            </div>
          );
        })}

        {decisionLoading && (
          <div style={{ width: '100%' }}>
            <div style={{
              border: '1px solid var(--color-border-tertiary)',
              borderRadius: 'var(--border-radius-md)',
              overflow: 'hidden',
            }}>
              <div style={{
                padding: '0.6rem 1rem',
                background: '#10B98120',
                color: '#10B981',
                fontSize: '13px', fontWeight: 600, letterSpacing: '0.02em',
              }}>
                Generating System Decision...
              </div>
              <div style={{
                padding: '1rem',
                background: 'var(--color-background-secondary)',
                display: 'flex', gap: '6px', alignItems: 'center',
                fontSize: '13px', color: 'var(--color-text-secondary)',
              }}>
                {[0, 0.2, 0.4].map((delay, i) => (
                  <div key={i} style={{
                    width: '8px', height: '8px', borderRadius: '50%',
                    background: 'var(--color-text-secondary)',
                    animation: `pulse 1.4s infinite ${delay}s`,
                  }} />
                ))}
                <span style={{ marginLeft: '4px' }}>Analyzing summary to generate decision...</span>
              </div>
            </div>
          </div>
        )}

        {isLoading && (
          <div style={{ display: 'flex', gap: '6px', alignItems: 'center', padding: '0.75rem 1rem' }}>
            {[0, 0.2, 0.4].map((delay, i) => (
              <div key={i} style={{
                width: '8px', height: '8px', borderRadius: '50%',
                background: 'var(--color-text-secondary)',
                animation: `pulse 1.4s infinite ${delay}s`,
              }} />
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input Area */}
      <div style={{
        padding: '1rem', borderTop: '0.5px solid var(--color-border-tertiary)',
        display: 'flex', flexDirection: 'column', gap: '8px',
      }}>
        {floorPlan && (
          <div style={{
            display: 'flex', alignItems: 'center', gap: '8px',
            padding: '6px 10px', background: 'var(--color-background-secondary)',
            borderRadius: 'var(--border-radius-md)', fontSize: '12px', color: 'var(--color-text-secondary)',
          }}>
            <span>📎 {floorPlan.name}</span>
            <button
              onClick={() => setFloorPlan(null)}
              style={{ marginLeft: 'auto', background: 'none', border: 'none', cursor: 'pointer', color: 'var(--color-text-secondary)', fontSize: '14px', lineHeight: 1, padding: '0 2px' }}
              title="Remove floor plan"
            >×</button>
          </div>
        )}

        <div style={{ display: 'flex', gap: '8px' }}>
          <input
            ref={floorPlanInputRef}
            type="file"
            accept="image/jpeg,image/png,image/gif,image/webp"
            onChange={handleFloorPlanUpload}
            style={{ display: 'none' }}
          />
          <button
            onClick={() => floorPlanInputRef.current?.click()}
            disabled={isLoading}
            title="Upload floor plan (required to start)"
            style={{
              padding: '0.75rem', borderRadius: 'var(--border-radius-md)',
              cursor: isLoading ? 'default' : 'pointer', fontSize: '16px', lineHeight: 1,
              background: conversationHistory.length === 0 ? 'var(--color-background-info)' : 'var(--color-background-secondary)',
              border: conversationHistory.length === 0 ? '1.5px solid var(--color-border-secondary)' : '0.5px solid var(--color-border-tertiary)',
              color: conversationHistory.length === 0 ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
            }}
          >&#128247;</button>

          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={conversationHistory.length === 0 ? "Upload a floor plan to start the conversation..." : "Describe your property or ask anything..."}
            disabled={isLoading || conversationHistory.length === 0}
            style={{
              flex: 1, padding: '0.75rem',
              border: '0.5px solid var(--color-border-tertiary)',
              borderRadius: 'var(--border-radius-md)', fontSize: '14px',
              fontFamily: 'inherit', resize: 'none', minHeight: '40px', maxHeight: '80px',
              color: 'var(--color-text-primary)', backgroundColor: 'var(--color-background-primary)',
            }}
            rows="1"
          />

          <button
            onClick={handleSend}
            disabled={!input.trim() || isLoading || conversationHistory.length === 0}
            style={{
              padding: '0.75rem 1.25rem',
              background: input.trim() && !isLoading ? 'var(--color-background-info)' : 'var(--color-background-secondary)',
              color: input.trim() && !isLoading ? 'var(--color-text-info)' : 'var(--color-text-secondary)',
              border: '0.5px solid var(--color-border-secondary)', borderRadius: 'var(--border-radius-md)',
              cursor: input.trim() && !isLoading ? 'pointer' : 'default',
              fontSize: '14px', fontWeight: 500, whiteSpace: 'nowrap', transition: 'all 0.2s',
            }}
          >Send</button>
        </div>
      </div>

      <style>{`
        @keyframes pulse {
          0%, 100% { opacity: 0.4; }
          50% { opacity: 1; }
        }
      `}</style>
    </div>
  );
}