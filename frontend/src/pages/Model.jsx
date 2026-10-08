import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getModelStatus, trainModel, exportModel } from '../api';
import { useToast } from '../hooks/useToast';

function StatCard({ label, value, icon, color = 'gray' }) {
  const colorMap = {
    gray: 'bg-gray-50 text-gray-700',
    blue: 'bg-primary-50 text-primary-700',
    green: 'bg-green-50 text-green-700',
    amber: 'bg-amber-50 text-amber-700',
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <div className="flex items-center gap-3">
        <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${colorMap[color]}`}>
          {icon}
        </div>
        <div>
          <p className="text-2xl font-bold text-gray-900">{value ?? '--'}</p>
          <p className="text-xs text-gray-500">{label}</p>
        </div>
      </div>
    </div>
  );
}

export default function Model() {
  const toast = useToast();
  const queryClient = useQueryClient();

  const { data: status, isLoading } = useQuery({
    queryKey: ['modelStatus'],
    queryFn: getModelStatus,
    refetchInterval: (query) => {
      // Poll more frequently while training
      if (query.state.data?.training === true || query.state.data?.status === 'training') {
        return 3000;
      }
      return false;
    },
  });

  const trainMutation = useMutation({
    mutationFn: trainModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['modelStatus'] });
      toast.success('Model training started');
    },
    onError: () => {
      toast.error('Failed to start training.');
    },
  });

  const exportMutation = useMutation({
    mutationFn: exportModel,
    onSuccess: () => {
      toast.success('Model exported and download started');
    },
    onError: () => {
      toast.error('Failed to export model.');
    },
  });

  const isTraining = status?.training === true || status?.status === 'training';
  const version = status?.version ?? status?.model_version ?? null;
  const totalCorrections = status?.total_corrections ?? status?.corrections_count ?? 0;
  const trainingCorrections = status?.training_corrections ?? status?.trained_on ?? 0;
  const accuracy = status?.accuracy ?? status?.accuracy_estimate ?? null;
  const lastTrained = status?.last_trained ?? status?.last_trained_at ?? null;

  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <h1 className="text-xl font-bold text-gray-900 mb-6">Model</h1>

      {isLoading ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            {[1, 2, 3, 4].map((i) => (
              <div key={i} className="skeleton h-20 rounded-xl" />
            ))}
          </div>
          <div className="skeleton h-40 rounded-xl" />
        </div>
      ) : (
        <>
          {/* Stats grid */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-6">
            <StatCard
              label="Model Version"
              value={version ?? 'v0'}
              color="blue"
              icon={
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
              }
            />
            <StatCard
              label="Total Corrections"
              value={totalCorrections}
              color="green"
              icon={
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                </svg>
              }
            />
            <StatCard
              label="Trained On"
              value={trainingCorrections}
              color="amber"
              icon={
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.387-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
                </svg>
              }
            />
            <StatCard
              label="Accuracy"
              value={accuracy !== null ? `${Math.round(accuracy * 100)}%` : 'N/A'}
              color="gray"
              icon={
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
                </svg>
              }
            />
          </div>

          {/* Training status card */}
          <div className="bg-white rounded-xl border border-gray-200 p-6 mb-6">
            <h2 className="text-sm font-semibold text-gray-700 mb-4">Training</h2>

            {isTraining ? (
              <div className="mb-4">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-5 h-5 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
                  <span className="text-sm font-medium text-primary-700">
                    Training in progress...
                  </span>
                </div>
                <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div className="h-full bg-primary-500 rounded-full animate-pulse" style={{ width: '60%' }} />
                </div>
                <p className="text-xs text-gray-400 mt-2">
                  The model is being fine-tuned with your corrections. This may take several minutes.
                </p>
              </div>
            ) : (
              <div className="mb-4">
                {lastTrained && (
                  <p className="text-sm text-gray-500 mb-2">
                    Last trained: {new Date(lastTrained).toLocaleString()}
                  </p>
                )}
                {totalCorrections > trainingCorrections ? (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 mb-3">
                    <p className="text-sm text-amber-800">
                      <strong>{totalCorrections - trainingCorrections}</strong> new corrections available since last training.
                      Train the model to improve accuracy.
                    </p>
                  </div>
                ) : totalCorrections === 0 ? (
                  <div className="bg-gray-50 border border-gray-200 rounded-lg p-3 mb-3">
                    <p className="text-sm text-gray-600">
                      No corrections yet. Use Play Mode to review OCR results and build training data.
                    </p>
                  </div>
                ) : (
                  <div className="bg-green-50 border border-green-200 rounded-lg p-3 mb-3">
                    <p className="text-sm text-green-800">
                      Model is up to date with all corrections.
                    </p>
                  </div>
                )}
              </div>
            )}

            <div className="flex flex-col sm:flex-row gap-3">
              <button
                onClick={() => trainMutation.mutate()}
                disabled={trainMutation.isPending || isTraining || totalCorrections === 0}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {trainMutation.isPending || isTraining ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    Training...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    Train Model
                  </>
                )}
              </button>

              <button
                onClick={() => exportMutation.mutate()}
                disabled={exportMutation.isPending || isTraining || !version}
                className="flex-1 flex items-center justify-center gap-2 px-4 py-2.5 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
              >
                {exportMutation.isPending ? (
                  <>
                    <div className="w-4 h-4 border-2 border-gray-400 border-t-transparent rounded-full animate-spin" />
                    Exporting...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
                    </svg>
                    Export Model
                  </>
                )}
              </button>
            </div>
          </div>

          {/* Info section */}
          <div className="bg-white rounded-xl border border-gray-200 p-6">
            <h2 className="text-sm font-semibold text-gray-700 mb-3">How it works</h2>
            <div className="space-y-3 text-sm text-gray-600">
              <div className="flex gap-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                  1
                </div>
                <p>Upload handwritten documents and run OCR to get initial text recognition.</p>
              </div>
              <div className="flex gap-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                  2
                </div>
                <p>Review and correct the OCR results using Play Mode. The more corrections you provide, the better the model gets.</p>
              </div>
              <div className="flex gap-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                  3
                </div>
                <p>Train the model with your corrections. Each training session fine-tunes the model to better recognize your handwriting.</p>
              </div>
              <div className="flex gap-3">
                <div className="w-6 h-6 bg-primary-100 text-primary-700 rounded-full flex items-center justify-center text-xs font-bold shrink-0">
                  4
                </div>
                <p>Export the trained model as a zip file to use it in other applications or for backup.</p>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
