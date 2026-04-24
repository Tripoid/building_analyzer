/// Streamed from [AnalysisRepository.analyze] so the loading screen can
/// reflect real byte-level progress instead of the old fake timer.
class UploadProgress {
  const UploadProgress({
    required this.sent,
    required this.total,
    required this.stage,
  });

  /// Bytes uploaded so far.
  final int sent;

  /// Total bytes to upload (may be -1 before headers are exchanged).
  final int total;

  /// Stage of the long-running analysis pipeline.
  final AnalysisStage stage;

  /// Fraction 0..1 of upload progress; stage gives the human label.
  double get ratio => total > 0 ? sent / total : 0;

  UploadProgress copyWith({int? sent, int? total, AnalysisStage? stage}) =>
      UploadProgress(
        sent: sent ?? this.sent,
        total: total ?? this.total,
        stage: stage ?? this.stage,
      );

  static const UploadProgress initial = UploadProgress(
    sent: 0,
    total: -1,
    stage: AnalysisStage.preparing,
  );
}

enum AnalysisStage {
  preparing('Подготовка фото'),
  uploading('Загрузка фото на сервер'),
  analyzing('Детекция дефектов и материалов'),
  estimating('Расчёт сметы в рублях'),
  done('Готово');

  const AnalysisStage(this.label);
  final String label;
}
