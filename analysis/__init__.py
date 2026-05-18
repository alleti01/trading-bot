"""Trade analysis + mistake-learning system.

Pipeline (per trade):

    PaperTradingLoop.on_trade_closed
        → TradeAnalyzer.build(...)        # deterministic
        → MistakeClassifier.classify(...) # deterministic, rule-based
        → persist to trade_analyses + trade_mistake_tags
        → PostTradeReport (markdown)
        → optional TradeAnalysisAgent (LLM, summary only)
        → NotificationService.notify("trade.analysis", …)

Pipeline (batch, end-of-day or operator-driven):

    PatternMiner.aggregate(...)            # stats by strategy / hour / regime / tag
    ImprovementSuggester.propose(...)      # validation_status="proposed" only
    FeedbackDataset.export(...)            # rows for offline retrain
    PromotionWorkflow.compare(...)         # candidate vs incumbent (walk-forward)

The system **never** auto-modifies strategy code, risk limits, model
thresholds, or model promotion. Every action that changes behavior is
either a logged proposal (analysis layer) or an explicit operator CLI
command (promotion).
"""
